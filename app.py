import streamlit as st
import pandas as pd
import numpy as np
import re
import logging
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import warnings
import time
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="智能彩票分析检测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 基础分析器架构 ====================
from abc import ABC, abstractmethod
from collections import defaultdict
import pandas as pd
import streamlit as st

class BaseAnalyzer(ABC):
    """基础分析器抽象类"""
    
    def __init__(self):
        self.content_parser = ContentParser()
        self.normalizer = PlayCategoryNormalizer()
        self.seen_records = set()
    
    @abstractmethod
    def analyze(self, df):
        """分析入口方法 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_supported_lottery_types(self):
        """返回支持的彩种类型"""
        pass
    
    def _get_record_hash(self, record):
        """生成记录的唯一哈希值"""
        key_parts = [
            record['会员账号'],
            record['彩种'], 
            record['期号'],
            record.get('玩法分类', ''),
            record.get('违规类型', ''),
            record.get('位置', ''),
            str(record.get('号码数量', 0)),
            record.get('矛盾类型', '')
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _add_unique_result(self, results, result_type, record):
        """添加唯一的结果记录"""
        record_hash = self._get_record_hash(record)
        
        if record_hash not in self.seen_records:
            self.seen_records.add(record_hash)
            results[result_type].append(record)
            return True
        return False
    
    def _calculate_sort_weight(self, record, result_type):
        """计算排序权重"""
        weight = 0
        
        # 基于号码数量
        if record.get('号码数量', 0) > 0:
            weight += record['号码数量'] * 10
        
        # 基于矛盾类型复杂度
        if record.get('矛盾类型'):
            conflict_count = len(record['矛盾类型'].split('、'))
            weight += conflict_count * 15
        
        # 基于其他数量字段
        for field in ['生肖数量', '尾数数量', '投注区间数', '投注波色数', '投注五行数']:
            if record.get(field, 0) > 0:
                weight += record[field] * 8
        
        # 基于矛盾值
        if record.get('矛盾值', 0) > 0:
            weight += record['矛盾值'] * 5
        
        # 基于检测类型重要性
        if '多号码' in result_type:
            weight += 25
        elif '矛盾' in result_type:
            weight += 20
        elif '全包' in result_type:
            weight += 30
        elif '三组' in result_type:
            weight += 35
        
        return weight

class PrecisionThresholdManager:
    """精确阈值管理器"""
    
    def get_threshold(self, lottery_type, play_category, detection_type):
        """获取精确阈值"""
        base_config = THRESHOLD_CONFIG.get(lottery_type, {})
        
        # 玩法特定的阈值配置
        play_specific_config = {
            'LHC': {
                '特码': {'multi_number': 31},
                '正码': {'multi_number': 25},
                '正1特': {'multi_number': 20},
                '正2特': {'multi_number': 20},
                '正3特': {'multi_number': 20},
                '正4特': {'multi_number': 20},
                '正5特': {'multi_number': 20},
                '正6特': {'multi_number': 20},
                '尾数': {'multi_tail': 7},
                '特尾': {'multi_tail': 5},
                '全尾': {'multi_tail': 5},
                '平特': {'zodiac_play': 7},
                '特肖': {'zodiac_play': 7},
                '一肖': {'zodiac_play': 7}
            },
            'K3': {
                '和值': {'hezhi_multi_number': 11},
                '独胆': {'dudan_multi_number': 5}
            },
            'PK10': {
                '冠军': {'multi_number': 8},
                '亚军': {'multi_number': 8},
                '第三名': {'multi_number': 8},
                '冠亚和': {'gyh_multi_number': 12}
            }
        }
        
        # 优先使用玩法特定配置
        play_config = play_specific_config.get(lottery_type, {}).get(play_category, {})
        if detection_type in play_config:
            return play_config[detection_type]
        
        # 回退到基础配置
        return base_config.get(detection_type, 5)  # 默认阈值

# ==================== 配置常量 ====================
LOTTERY_CONFIGS = {
    'PK10': {
        'lotteries': [
            '分分PK拾', '三分PK拾', '五分PK拾', '新幸运飞艇', '澳洲幸运10',
            '一分PK10', '宾果PK10', '极速飞艇', '澳洲飞艇', '幸运赛车',
            '分分赛车', '北京PK10', '旧北京PK10', '极速赛车', '幸运赛車', 
            '北京赛车', '极速PK10', '幸运PK10', '赛车', '赛車'
        ],
        'min_number': 1,
        'max_number': 10,
        'gyh_min': 3,
        'gyh_max': 19,
        'position_names': ['冠军', '亚军', '第三名', '第四名', '第五名', 
                          '第六名', '第七名', '第八名', '第九名', '第十名']
    },
    'K3': {
        'lotteries': [
            '分分快三', '三分快3', '五分快3', '澳洲快三', '宾果快三',
            '1分快三', '3分快三', '5分快三', '10分快三', '加州快三',
            '幸运快三', '大发快三', '快三', '快3', 'k3', 'k三', 
            '澳门快三', '香港快三', '江苏快三'
        ],
        'min_number': 1,
        'max_number': 6,
        'hezhi_min': 3,
        'hezhi_max': 18
    },
    'LHC': {
        'lotteries': [
            '新澳门六合彩', '澳门六合彩', '香港六合彩', '一分六合彩',
            '五分六合彩', '三分六合彩', '香港⑥合彩', '分分六合彩',
            '快乐6合彩', '港⑥合彩', '台湾大乐透', '六合', 'lhc', '六合彩',
            '⑥合', '6合', '大发六合彩'
        ],
        'min_number': 1,
        'max_number': 49
    },
    '3D': {
        'lotteries': [
            '排列三', '排列3', '幸运排列3', '一分排列3', '二分排列3', '三分排列3', 
            '五分排列3', '十分排列3', '大发排列3', '好运排列3', '福彩3D', '极速3D',
            '极速排列3', '幸运3D', '一分3D', '二分3D', '三分3D', '五分3D', 
            '十分3D', '大发3D', '好运3D'
        ],
        'min_number': 0,
        'max_number': 9,
        'dingwei_threshold': 7  # 定位胆多码阈值
    },
    'SSC': {
        'lotteries': [
            '分分时时彩', '三分时时彩', '五分时时彩', '宾果时时彩',
            '1分时时彩', '3分时时彩', '5分时时彩', '旧重庆时时彩',
            '幸运时时彩', '腾讯分分彩', '新疆时时彩', '天津时时彩',
            '重庆时时彩', '上海时时彩', '广东时时彩', '分分彩', '时时彩', '時時彩'
        ],
        'min_number': 0,
        'max_number': 9
    },
    'THREE_COLOR': {
        'lotteries': [
            '一分三色彩', '30秒三色彩', '五分三色彩', '三分三色彩',
            '三色', '三色彩', '三色球'
        ],
        'min_number': 0,
        'max_number': 9
    }
}

THRESHOLD_CONFIG = {
    'PK10': {
        'multi_number': 8,
        'gyh_multi_number': 12,
        'position_multi': 8,
        'all_positions_bet': 10
    },
    'K3': {
        'multi_number': 5,
        'hezhi_multi_number': 13,
        'value_size_contradiction': 5,
        'dudan_multi_number': 5
    },
    'LHC': {
        'number_play': 31,
        'zodiac_play': 7,
        'tail_play': 7,
        'range_bet': 4,
        'lianxiao_threshold': 7,
        'lianwei_threshold': 7,
        'wave_bet': 3,
        'five_elements': 4
    },
    '3D': {
        'dingwei_multi': 7,  # 定位胆多码阈值
        'two_sides_conflict': 2  # 两面矛盾检测
    },
    'SSC': {
        'dingwei_multi': 8,
        'douniu_multi': 8,
        'two_sides_conflict': 2
    },
    'THREE_COLOR': {
        'zhengma_multi': 7,
        'two_sides_conflict': 2,
        'wave_conflict': 2
    }
}

# ==================== 新增：精确阈值管理器 ====================
class PrecisionThresholdManager:
    """精确阈值管理器"""
    
    def get_threshold(self, lottery_type, play_category, detection_type):
        """获取精确阈值"""
        base_config = THRESHOLD_CONFIG.get(lottery_type, {})
        
        # 玩法特定的阈值配置
        play_specific_config = {
            'LHC': {
                '特码': {'multi_number': 31},
                '正码': {'multi_number': 25},
                '正1特': {'multi_number': 20},
                '正2特': {'multi_number': 20},
                '正3特': {'multi_number': 20},
                '正4特': {'multi_number': 20},
                '正5特': {'multi_number': 20},
                '正6特': {'multi_number': 20},
                '尾数': {'multi_tail': 7},
                '特尾': {'multi_tail': 5},
                '全尾': {'multi_tail': 5},
                '平特': {'zodiac_play': 7},
                '特肖': {'zodiac_play': 7},
                '一肖': {'zodiac_play': 7}
            },
            'K3': {
                '和值': {'hezhi_multi_number': 11},
                '独胆': {'dudan_multi_number': 5}
            },
            'PK10': {
                '冠军': {'multi_number': 8},
                '亚军': {'multi_number': 8},
                '第三名': {'multi_number': 8},
                '冠亚和': {'gyh_multi_number': 12}
            }
        }
        
        # 优先使用玩法特定配置
        play_config = play_specific_config.get(lottery_type, {}).get(play_category, {})
        if detection_type in play_config:
            return play_config[detection_type]
        
        # 回退到基础配置
        return base_config.get(detection_type, 5)  # 默认阈值

# ==================== 日志设置 ====================
def setup_logging():
    """设置日志系统"""
    logger = logging.getLogger('LotteryAnalysis')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ==================== 数据处理类 ====================
class DataProcessor:
    def __init__(self):
        self.required_columns = ['会员账号', '彩种', '期号', '玩法', '内容', '金额']
        self.column_mapping = {
            '会员账号': ['会员账号', '会员账户', '账号', '账户', '用户账号', '玩家账号', '用户ID', '玩家ID'],
            '彩种': ['彩种', '彩神', '彩票种类', '游戏类型', '彩票类型', '游戏彩种', '彩票名称'],
            '期号': ['期号', '期数', '期次', '期', '奖期', '期号信息', '期号编号'],
            '玩法': ['玩法', '玩法分类', '投注类型', '类型', '投注玩法', '玩法类型', '分类'],
            '内容': ['内容', '投注内容', '下注内容', '注单内容', '投注号码', '号码内容', '投注信息'],
            '金额': ['金额', '下注总额', '投注金额', '总额', '下注金额', '投注额', '金额数值']
        }
    
    def smart_column_identification(self, df_columns):
        """智能列识别"""
        identified_columns = {}
        actual_columns = [str(col).strip() for col in df_columns]
        
        with st.expander("🔍 列名识别详情", expanded=False):
            st.info(f"检测到的列名: {actual_columns}")
            
            for standard_col, possible_names in self.column_mapping.items():
                found = False
                for actual_col in actual_columns:
                    actual_col_lower = actual_col.lower().replace(' ', '').replace('_', '').replace('-', '')
                    
                    for possible_name in possible_names:
                        possible_name_lower = possible_name.lower().replace(' ', '').replace('_', '').replace('-', '')
                        
                        # 增强会员账号识别
                        if standard_col == '会员账号':
                            # 更宽松的匹配规则
                            account_keywords = ['会员', '账号', '账户', '用户', '玩家', 'id']
                            if any(keyword in actual_col_lower for keyword in account_keywords):
                                identified_columns[actual_col] = standard_col
                                st.success(f"✅ 识别列名: {actual_col} -> {standard_col}")
                                found = True
                                break
                        else:
                            # 其他列的原有匹配逻辑
                            if (possible_name_lower in actual_col_lower or 
                                actual_col_lower in possible_name_lower or
                                len(set(possible_name_lower) & set(actual_col_lower)) / len(possible_name_lower) > 0.7):
                                identified_columns[actual_col] = standard_col
                                st.success(f"✅ 识别列名: {actual_col} -> {standard_col}")
                                found = True
                                break
                    
                    if found:
                        break
                
                if not found:
                    st.warning(f"⚠️ 未识别到 {standard_col} 对应的列名")
        
        return identified_columns
    
    def find_data_start(self, df):
        """智能找到数据起始位置"""
        for row_idx in range(min(20, len(df))):
            for col_idx in range(min(10, len(df.columns))):
                cell_value = str(df.iloc[row_idx, col_idx])
                if pd.notna(cell_value) and any(keyword in cell_value for keyword in ['会员', '账号', '期号', '彩种', '玩法', '内容', '订单', '用户']):
                    return row_idx, col_idx
        return 0, 0
    
    def validate_data_quality(self, df):
        """数据质量验证"""
        logger.info("正在进行数据质量验证...")
        issues = []
        
        # 检查必要列
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"缺少必要列: {missing_cols}")
        
        # 检查空值
        for col in self.required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    issues.append(f"列 '{col}' 有 {null_count} 个空值")
        
        # 特别检查会员账号的完整性
        if '会员账号' in df.columns:
            # 检查是否有被截断的账号
            truncated_accounts = df[df['会员账号'].str.contains(r'\.\.\.|…', na=False)]
            if len(truncated_accounts) > 0:
                issues.append(f"发现 {len(truncated_accounts)} 个可能被截断的会员账号")
            
            # 检查账号长度异常的情况
            account_lengths = df['会员账号'].str.len()
            if account_lengths.max() > 50:  # 假设正常账号长度不超过50个字符
                issues.append("发现异常长度的会员账号")
            
            # 显示账号格式样本
            unique_accounts = df['会员账号'].unique()[:5]
            sample_info = " | ".join([f"'{acc}'" for acc in unique_accounts])
            st.info(f"会员账号格式样本: {sample_info}")
        
        # 检查数据类型
        if '期号' in df.columns:
            # 修复期号格式问题：去掉.0
            df['期号'] = df['期号'].astype(str).str.replace(r'\.0$', '', regex=True)
            # 允许期号包含字母和数字
            invalid_periods = df[~df['期号'].str.match(r'^[\dA-Za-z]+$')]
            if len(invalid_periods) > 0:
                issues.append(f"发现 {len(invalid_periods)} 条无效期号记录")
        
        # 检查重复数据
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"发现 {duplicate_count} 条重复记录")
        
        if issues:
            with st.expander("⚠️ 数据质量问题", expanded=True):
                for issue in issues:
                    st.warning(f"  - {issue}")
        else:
            st.success("✅ 数据质量检查通过")
        
        return issues
    
    def clean_data(self, uploaded_file):
        """数据清洗主函数"""
        try:
            # 第一次读取用于定位
            df_temp = pd.read_excel(uploaded_file, header=None, nrows=50)
            st.info(f"原始数据维度: {df_temp.shape}")
            
            # 找到数据起始位置
            start_row, start_col = self.find_data_start(df_temp)
            st.info(f"数据起始位置: 第{start_row+1}行, 第{start_col+1}列")
            
            # 重新读取数据 - 特别处理常规格式单元格
            df_clean = pd.read_excel(
                uploaded_file, 
                header=start_row,
                skiprows=range(start_row + 1) if start_row > 0 else None,
                dtype=str,  # 将所有列读取为字符串
                na_filter=False,  # 不过滤空值
                keep_default_na=False,  # 不使用默认的NA值处理
                converters={}  # 为空，让pandas不要进行任何转换
            )
            
            # 删除起始列之前的所有列
            if start_col > 0:
                df_clean = df_clean.iloc[:, start_col:]
            
            st.info(f"清理后数据维度: {df_clean.shape}")
            
            # 智能列识别
            column_mapping = self.smart_column_identification(df_clean.columns)
            if column_mapping:
                df_clean = df_clean.rename(columns=column_mapping)
                st.success("✅ 列名识别完成!")
                for old_col, new_col in column_mapping.items():
                    logger.info(f"  {old_col} -> {new_col}")
            
            # 确保必要列存在
            missing_columns = [col for col in self.required_columns if col not in df_clean.columns]
            if missing_columns and len(df_clean.columns) >= 4:
                st.warning("自动映射列名...")
                manual_mapping = {}
                col_names = ['会员账号', '彩种', '期号', '内容', '玩法', '金额']
                for i, col_name in enumerate(col_names):
                    if i < len(df_clean.columns):
                        manual_mapping[df_clean.columns[i]] = col_name
                
                df_clean = df_clean.rename(columns=manual_mapping)
                st.info(f"手动重命名后的列: {list(df_clean.columns)}")
            
            # 数据清理
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[col for col in self.required_columns if col in df_clean.columns])
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # 数据类型转换 - 特别小心处理会员账号
            for col in self.required_columns:
                if col in df_clean.columns:
                    if col == '会员账号':
                        # 特别处理会员账号：确保不丢失任何字符
                        df_clean[col] = df_clean[col].apply(
                            lambda x: str(x) if pd.notna(x) else ''
                        )
                    else:
                        df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # 修复期号格式：去掉.0
            if '期号' in df_clean.columns:
                df_clean['期号'] = df_clean['期号'].str.replace(r'\.0$', '', regex=True)
            
            # 数据质量验证 - 添加会员账号完整性检查
            self.validate_data_quality(df_clean)
            
            st.success(f"✅ 数据清洗完成: {initial_count} -> {len(df_clean)} 条记录")
            
            # 在 clean_data 方法中，修改显示会员账号样本的部分：
                
            st.info(f"📊 唯一会员账号数: {df_clean['会员账号'].nunique()}")
            
            # 彩种分布显示
            lottery_dist = df_clean['彩种'].value_counts()
            with st.expander("🎯 彩种分布", expanded=False):
                st.dataframe(lottery_dist.reset_index().rename(columns={'index': '彩种', '彩种': '数量'}))
            
            return df_clean
            
        except Exception as e:
            st.error(f"❌ 数据清洗失败: {str(e)}")
            logger.error(f"数据清洗失败: {str(e)}")
            return None

# ==================== 内容解析器 ====================
class ContentParser:
    """统一的投注内容解析器"""

    @staticmethod
    def parse_pk10_vertical_format(content):
        """
        解析PK10竖线分隔的定位胆格式
        格式：号码1,号码2|号码3|号码4,号码5|号码6|号码7,号码8,号码9|号码10
        或者：_|05|_|_|_ 表示只有第二个位置有投注
        """
        content_str = str(content).strip()
        bets_by_position = defaultdict(list)
        
        if not content_str:
            return bets_by_position
        
        # 定义位置映射
        positions = ['冠军', '亚军', '第三名', '第四名', '第五名', 
                    '第六名', '第七名', '第八名', '第九名', '第十名']
        
        # 按竖线分割
        parts = content_str.split('|')
        
        for i, part in enumerate(parts):
            if i < len(positions):
                position = positions[i]
                part_clean = part.strip()
                
                # 跳过空位或下划线
                if not part_clean or part_clean == '_' or part_clean == '':
                    continue
                
                # 提取数字（可能是单个数字或多个逗号分隔的数字）
                numbers = []
                if ',' in part_clean:
                    # 逗号分隔的多个数字
                    number_strs = part_clean.split(',')
                    for num_str in number_strs:
                        num_clean = num_str.strip()
                        if num_clean.isdigit():
                            numbers.append(int(num_clean))
                else:
                    # 单个数字
                    if part_clean.isdigit():
                        numbers.append(int(part_clean))
                
                # 添加到对应位置
                bets_by_position[position].extend(numbers)
        
        return bets_by_position

    @staticmethod
    def parse_ssc_vertical_format(content):
        """
        解析时时彩竖线分隔的定位胆格式
        格式：号码1,号码2|号码3|号码4,号码5|号码6|号码7,号码8,号码9|号码10
        或者：_|05|_|_|_ 表示只有第二个位置有投注
        """
        content_str = str(content).strip()
        bets_by_position = defaultdict(list)
        
        if not content_str:
            return bets_by_position
        
        # 定义位置映射
        positions = ['第1球', '第2球', '第3球', '第4球', '第5球']
        
        # 按竖线分割
        parts = content_str.split('|')
        
        for i, part in enumerate(parts):
            if i < len(positions):
                position = positions[i]
                part_clean = part.strip()
                
                # 跳过空位或下划线
                if not part_clean or part_clean == '_' or part_clean == '':
                    continue
                
                # 提取数字（可能是单个数字或多个逗号分隔的数字）
                numbers = []
                if ',' in part_clean:
                    # 逗号分隔的多个数字
                    number_strs = part_clean.split(',')
                    for num_str in number_strs:
                        num_clean = num_str.strip()
                        if num_clean.isdigit():
                            numbers.append(int(num_clean))
                else:
                    # 单个数字
                    if part_clean.isdigit():
                        numbers.append(int(part_clean))
                
                # 添加到对应位置
                bets_by_position[position].extend(numbers)
        
        return bets_by_position
    
    @staticmethod
    def parse_positional_bets(content, position_keywords=None):
        """
        解析位置投注内容
        格式：位置1-投注项1,投注项2,位置2-投注项1,投注项2,...
        """
        content_str = str(content).strip()
        bets_by_position = defaultdict(list)
        
        if not content_str:
            return bets_by_position
        
        # 按逗号分割所有部分
        parts = [part.strip() for part in content_str.split(',')]
        
        current_position = None
        
        for part in parts:
            # 检查是否包含位置关键词
            is_position = False
            if position_keywords:
                for keyword in position_keywords:
                    if keyword in part and '-' in part:
                        is_position = True
                        break
            
            # 如果包含位置信息或者是明确的"位置-内容"格式
            if '-' in part and (is_position or position_keywords is None):
                try:
                    position_part, bet_value = part.split('-', 1)
                    current_position = position_part.strip()
                    bets_by_position[current_position].append(bet_value.strip())
                except ValueError:
                    # 分割失败，可能不是有效的位置格式
                    if current_position:
                        bets_by_position[current_position].append(part)
            elif current_position:
                # 属于当前位置的投注项
                bets_by_position[current_position].append(part)
            else:
                # 没有当前位置，可能是独立的投注项
                bets_by_position['未知位置'].append(part)
        
        return bets_by_position
    
    @staticmethod
    def parse_pk10_content(content):
        """解析PK10投注内容 - 增强版，支持竖线格式"""
        pk10_positions = ['冠军', '亚军', '第三名', '第四名', '第五名', 
                         '第六名', '第七名', '第八名', '第九名', '第十名',
                         '第1名', '第2名', '第3名', '第4名', '第5名',
                         '第6名', '第7名', '第8名', '第9名', '第10名',
                         '前一', '前二', '前三']
        
        content_str = str(content).strip()
        
        # 首先检查是否是竖线分隔格式
        if '|' in content_str and any(char.isdigit() or char == '_' or char == ',' for char in content_str):
            vertical_result = ContentParser.parse_pk10_vertical_format(content_str)
            if any(vertical_result.values()):  # 如果有解析结果
                return vertical_result
        
        # 特殊处理"位置:号码"格式
        if ':' in content_str and re.search(r'\d{2}', content_str):
            match = re.match(r'^(.+?):([\d,]+)$', content_str)
            if match:
                position = match.group(1).strip()
                numbers_str = match.group(2)
                bets_by_position = defaultdict(list)
                
                normalized_position = position
                if '九' in position or '9' in position:
                    normalized_position = '第九名'
                
                numbers = re.findall(r'\d{2}', numbers_str)
                bets_by_position[normalized_position].extend([int(num) for num in numbers])
                return bets_by_position
        
        # 原有的解析逻辑
        return ContentParser.parse_positional_bets(content, pk10_positions)
    
    @staticmethod
    def parse_lhc_zhengma_content(content):
        """
        解析六合彩正码投注内容 - 增强版本
        格式：位置1-投注项1,投注项2,位置2-投注项1,投注项2,...
        """
        content_str = str(content).strip()
        bets_by_position = defaultdict(list)
        
        if not content_str:
            return bets_by_position
        
        # 按逗号分割所有部分
        parts = [part.strip() for part in content_str.split(',')]
        
        current_position = None
        
        for part in parts:
            # 检查是否包含位置关键词
            is_position = False
            position_keywords = ['正码一', '正码二', '正码三', '正码四', '正码五', '正码六',
                               '正1', '正2', '正3', '正4', '正5', '正6',
                               '正码1', '正码2', '正码3', '正码4', '正码5', '正码6']
            
            for keyword in position_keywords:
                if keyword in part and '-' in part:
                    is_position = True
                    break
            
            # 如果包含位置信息或者是明确的"位置-内容"格式
            if '-' in part and is_position:
                try:
                    position_part, bet_value = part.split('-', 1)
                    current_position = position_part.strip()
                    bets_by_position[current_position].append(bet_value.strip())
                except ValueError:
                    # 分割失败，可能不是有效的位置格式
                    if current_position:
                        bets_by_position[current_position].append(part)
            elif current_position:
                # 属于当前位置的投注项
                bets_by_position[current_position].append(part)
            else:
                # 没有当前位置，可能是独立的投注项
                bets_by_position['未知位置'].append(part)
        
        return bets_by_position
    
    @staticmethod
    def parse_ssc_content(content):
        """解析时时彩投注内容 - 增强竖线格式支持"""
        ssc_positions = ['第1球', '第2球', '第3球', '第4球', '第5球',
                        '万位', '千位', '百位', '十位', '个位']
        
        content_str = str(content).strip()
        
        # 首先检查是否是竖线分隔格式
        if '|' in content_str and any(char.isdigit() or char == '_' or char == ',' for char in content_str):
            vertical_result = ContentParser.parse_ssc_vertical_format(content_str)
            if any(vertical_result.values()):  # 如果有解析结果
                return vertical_result
        
        # 原有的解析逻辑
        return ContentParser.parse_positional_bets(content, ssc_positions)

    @staticmethod
    def parse_3d_vertical_format(content):
        """
        解析3D/排列3竖线分隔的定位胆格式
        格式：号码1,号码2|号码3|号码4,号码5,号码6
        或者：_|05|_ 表示只有第二个位置有投注
        """
        content_str = str(content).strip()
        bets_by_position = defaultdict(list)
        
        if not content_str:
            return bets_by_position
        
        # 定义位置映射 - 3D通常是百位、十位、个位
        positions = ['百位', '十位', '个位']
        
        # 按竖线分割
        parts = content_str.split('|')
        
        for i, part in enumerate(parts):
            if i < len(positions):
                position = positions[i]
                part_clean = part.strip()
                
                # 跳过空位或下划线
                if not part_clean or part_clean == '_' or part_clean == '':
                    continue
                
                # 提取数字（可能是单个数字或多个逗号分隔的数字）
                numbers = []
                if ',' in part_clean:
                    # 逗号分隔的多个数字
                    number_strs = part_clean.split(',')
                    for num_str in number_strs:
                        num_clean = num_str.strip()
                        if num_clean.isdigit():
                            numbers.append(int(num_clean))
                else:
                    # 单个数字
                    if part_clean.isdigit():
                        numbers.append(int(num_clean))
                
                # 添加到对应位置
                bets_by_position[position].extend(numbers)
        
        return bets_by_position

    @staticmethod
    def infer_position_from_content(content, lottery_type):
        """从内容和彩种类型推断位置"""
        content_str = str(content)
        
        if lottery_type == 'PK10':
            # PK10位置推断逻辑
            pk10_positions = {
                '冠军': ['冠军', '第1名', '第一名', '前一'],
                '亚军': ['亚军', '第2名', '第二名'],
                '第三名': ['第三名', '季军', '第3名'],
                '第四名': ['第四名', '第4名'],
                '第五名': ['第五名', '第5名'],
                '第六名': ['第六名', '第6名'],
                '第七名': ['第七名', '第7名'],
                '第八名': ['第八名', '第8名'],
                '第九名': ['第九名', '第9名'],
                '第十名': ['第十名', '第10名']
            }
            for position, keywords in pk10_positions.items():
                for keyword in keywords:
                    if keyword in content_str:
                        return position
        
        elif lottery_type == 'SSC':
            # 时时彩位置推断逻辑
            ssc_positions = {
                '第1球': ['第1球', '万位', '第一位'],
                '第2球': ['第2球', '千位', '第二位'],
                '第3球': ['第3球', '百位', '第三位'],
                '第4球': ['第4球', '十位', '第四位'],
                '第5球': ['第5球', '个位', '第五位']
            }
            for position, keywords in ssc_positions.items():
                for keyword in keywords:
                    if keyword in content_str:
                        return position
        
        elif lottery_type == 'LHC':
            # 六合彩位置推断逻辑
            lhc_positions = {
                '正码1': ['正码一', '正1', '正码1'],
                '正码2': ['正码二', '正2', '正码2'],
                '正码3': ['正码三', '正3', '正码3'],
                '正码4': ['正码四', '正4', '正码4'],
                '正码5': ['正码五', '正5', '正码5'],
                '正码6': ['正码六', '正6', '正码6']
            }
            for position, keywords in lhc_positions.items():
                for keyword in keywords:
                    if keyword in content_str:
                        return position
        
        return '未知位置'

# ==================== 数据分析类 ====================
class DataAnalyzer:
    def __init__(self):
        self.cache = {}
        self.content_parser = ContentParser()  # 添加统一解析器
    
    @lru_cache(maxsize=10000)
    def extract_numbers_cached(self, content, min_num, max_num, is_pk10=False):
        """带缓存的号码提取函数"""
        return self.extract_numbers_from_content(content, min_num, max_num, is_pk10)
    
    def extract_numbers_from_content(self, content, min_num=0, max_num=49, is_pk10=False):
        """从内容中提取数字 - 增强三军格式处理"""
        numbers = []
        content_str = str(content)
        
        try:
            # 特殊处理三军格式：1,2,3,4,5,6
            if re.match(r'^(\d,)*\d$', content_str.strip()):
                numbers = [int(x.strip()) for x in content_str.split(',') if x.strip().isdigit()]
                # 过滤范围
                numbers = [num for num in numbers if min_num <= num <= max_num]
                return list(set(numbers))
            
            if is_pk10:
                # PK拾/赛车特殊处理：过滤掉"第X名"等玩法描述
                content_str = re.sub(r'第\d+名-?', '', content_str)
            
            # 提取数字
            number_matches = re.findall(r'\b\d{1,2}\b', content_str)
            for match in number_matches:
                num = int(match)
                if min_num <= num <= max_num:
                    numbers.append(num)
            
            return list(set(numbers))
        except Exception as e:
            logger.warning(f"号码提取失败: {content}, 错误: {str(e)}")
            return []
    
    def extract_zodiacs_from_content(self, content):
        """从内容中提取生肖"""
        zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        found_zodiacs = []
        
        content_str = str(content)
        for zodiac in zodiacs:
            if zodiac in content_str:
                found_zodiacs.append(zodiac)
        
        return list(set(found_zodiacs))
    
    def extract_tails_from_content(self, content):
        """从内容中提取尾数（连尾专用）"""
        tails = []
        content_str = str(content)
        
        # 匹配尾数模式：尾0、尾1、0尾、1尾等
        tail_patterns = [
            r'尾([0-9])',  # 尾0,尾1,...,尾9
            r'([0-9])尾',  # 0尾,1尾,...,9尾
        ]
        
        for pattern in tail_patterns:
            matches = re.findall(pattern, content_str)
            tails.extend([int(tail) for tail in matches])
        
        return list(set(tails))
    
    def extract_size_parity_from_content(self, content):
        """从内容中提取大小单双本"""
        content_str = str(content)
        size_parity = []
        
        # 使用更精确的匹配，避免误匹配
        if re.search(r'(?<!合)大(?![小尾])', content_str) or '特大' in content_str:
            size_parity.append('大')
        if re.search(r'(?<!合)小(?![大尾])', content_str) or '特小' in content_str:
            size_parity.append('小')
        if re.search(r'(?<!合)单(?![双])', content_str) or '特单' in content_str:
            size_parity.append('单')
        if re.search(r'(?<!合)双(?![单])', content_str) or '特双' in content_str:
            size_parity.append('双')
        
        return list(set(size_parity))
    
    def extract_dragon_tiger_from_content(self, content):
        """从内容中提取龙虎"""
        content_str = str(content)
        dragon_tiger = []
        
        if '龙' in content_str and '虎' not in content_str:
            dragon_tiger.append('龙')
        if '虎' in content_str and '龙' not in content_str:
            dragon_tiger.append('虎')
        
        return list(set(dragon_tiger))
    
    def extract_wave_color_from_content(self, content):
        """从内容中提取波色 - 增强版，支持半波项识别"""
        content_str = str(content)
        found_waves = []
        
        # 波色映射（包括七色波的所有颜色）
        wave_mappings = {
            '红波': ['红波', '紅色波', '红'],
            '蓝波': ['蓝波', '藍波', '蓝', '藍'],
            '绿波': ['绿波', '綠波', '绿', '綠'],
            '紫波': ['紫波', '紫'],
            '橙波': ['橙波', '橙'],
            '黄波': ['黄波', '黃波', '黄', '黃'],
            '青波': ['青波', '青']
        }
        
        for wave_name, keywords in wave_mappings.items():
            for keyword in keywords:
                if keyword in content_str:
                    # 检查是否是复合投注，如"红波-红双"
                    if '-' in content_str and f"{keyword}-" in content_str:
                        # 这种情况"红波"是玩法部分，不是实际投注内容
                        pass  # 添加pass语句，避免空的if分支
                    else:
                        # 检查是否被半波项包含（如"红大"包含"红"，但不是我们要的波色）
                        is_banbo_item = False
                        banbo_indicators = ['大', '小', '单', '双']
                        for indicator in banbo_indicators:
                            if f"{keyword}{indicator}" in content_str or f"{keyword} {indicator}" in content_str:
                                is_banbo_item = True
                                break
                        
                        if not is_banbo_item:
                            found_waves.append(wave_name)
                    break  # 找到一个关键词就跳出内层循环
        
        return list(set(found_waves))

    def extract_three_color_wave_from_content(self, content):
        """从内容中提取三色彩的波色 - 只提取红波、绿波、紫波"""
        content_str = str(content)
        found_waves = []
        
        # 处理繁体字和简体字
        if '红波' in content_str or '紅波' in content_str:
            found_waves.append('红波')
        if '绿波' in content_str or '綠波' in content_str:
            found_waves.append('绿波')
        if '紫波' in content_str:
            found_waves.append('紫波')
        
        return list(set(found_waves))
    
    def extract_five_elements_from_content(self, content):
        """从内容中提取五行"""
        content_str = str(content)
        elements = ['金', '木', '水', '火', '土']
        found_elements = []
        
        for element in elements:
            if element in content_str:
                found_elements.append(element)
        
        return list(set(found_elements))
    
    def extract_douniu_types(self, content):
        """提取斗牛类型"""
        content_str = str(content)
        bull_types = []
        
        # 移除"斗牛-"前缀
        clean_content = content_str.replace('斗牛-', '')
        
        # 斗牛类型列表
        all_types = ['无牛', '牛一', '牛二', '牛三', '牛四', '牛五', 
                    '牛六', '牛七', '牛八', '牛九', '牛牛']
        
        for bull_type in all_types:
            if bull_type in clean_content:
                bull_types.append(bull_type)
        
        return list(set(bull_types))
    
    def parse_pk10_gyh_content(self, content):
        """解析PK10冠亚和玩法内容"""
        content_str = str(content)
        result = {
            'numbers': set(),    # 和值号码
            'size_parity': set() # 大小单双
        }
        
        # 提取号码（3-19）
        numbers = re.findall(r'\b(1[0-9]|[3-9])\b', content_str)
        result['numbers'].update([int(num) for num in numbers])
        
        # 提取大小单双
        content_lower = content_str.lower()
        if '大' in content_lower or '冠亚大' in content_lower:
            result['size_parity'].add('大')
        if '小' in content_lower or '冠亚小' in content_lower:
            result['size_parity'].add('小')
        if '单' in content_lower or '冠亚单' in content_lower:
            result['size_parity'].add('单')
        if '双' in content_lower or '冠亚双' in content_lower:
            result['size_parity'].add('双')
        
        return result
    
    def parse_pk10_number_content(self, content):
        """解析PK10号码类玩法内容 - 增强竖线格式支持"""
        content_str = str(content)
        numbers_by_position = defaultdict(list)
        
        # 首先尝试竖线分隔格式
        if '|' in content_str and any(char.isdigit() or char == '_' or char == ',' for char in content_str):
            vertical_result = ContentParser.parse_pk10_vertical_format(content_str)
            if any(vertical_result.values()):
                return vertical_result
        
        # 处理竖线分隔的格式：01,02,03,04,05|07,08,06,09,10|...
        if '|' in content_str and re.search(r'\d{2}', content_str):
            positions = ['冠军', '亚军', '第三名', '第四名', '第五名']
            parts = content_str.split('|')
            
            for i, part in enumerate(parts):
                if i < len(positions):
                    position = positions[i]
                    numbers = re.findall(r'\d{2}', part)
                    numbers_by_position[position].extend([int(num) for num in numbers])
        
        # 处理"第九名:01,02,05,06,07,08,09,03"这种格式
        elif ':' in content_str and re.search(r'\d{2}', content_str):
            match = re.match(r'^(.+?):([\d,]+)$', content_str)
            if match:
                position = match.group(1).strip()
                numbers_str = match.group(2)
                position = self._normalize_pk10_position(position)
                if position:
                    numbers = re.findall(r'\d{2}', numbers_str)
                    numbers_by_position[position].extend([int(num) for num in numbers])
            else:
                parts = content_str.split(',')
                for part in parts:
                    if ':' in part:
                        position, numbers_str = part.split(':', 1)
                        position = self._normalize_pk10_position(position)
                        if position:
                            numbers = re.findall(r'\d{2}', numbers_str)
                            numbers_by_position[position].extend([int(num) for num in numbers])
        
        # 处理冠军-01,02,03格式
        elif '-' in content_str and re.search(r'\d{2}', content_str):
            parts = content_str.split(',')
            for part in parts:
                if '-' in part:
                    position, numbers_str = part.split('-', 1)
                    position = self._normalize_pk10_position(position)
                    numbers = re.findall(r'\d{2}', numbers_str)
                    numbers_by_position[position].extend([int(num) for num in numbers])
        
        # 处理纯数字格式
        else:
            numbers = self.extract_numbers_from_content(content_str, 1, 10, is_pk10=True)
            if numbers:
                position = self._infer_pk10_position_from_content(content_str)
                numbers_by_position[position].extend(numbers)
        
        # 去重
        for position in numbers_by_position:
            numbers_by_position[position] = list(set(numbers_by_position[position]))
        
        return numbers_by_position
    
    def _infer_pk10_position_from_content(self, content):
        """推断PK10位置"""
        content_str = str(content)
        
        position_mapping = {
            '冠军': ['冠军', '第1名', '第一名'],
            '亚军': ['亚军', '第2名', '第二名'],
            '第三名': ['第三名', '第3名', '季军'],
            '第四名': ['第四名', '第4名'],
            '第五名': ['第五名', '第5名'],
            '第六名': ['第六名', '第6名'],
            '第七名': ['第七名', '第7名'],
            '第八名': ['第八名', '第8名'],
            '第九名': ['第九名', '第9名'],
            '第十名': ['第十名', '第10名']
        }
        
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                if keyword in content_str:
                    return position
        
        return '冠军'
    
    def _normalize_pk10_position(self, position):
        """增强的PK10位置标准化"""
        position_mapping = {
            '冠军': '冠军', '第1名': '冠军', '第一名': '冠军', '1': '冠军', '1st': '冠军',
            '前一': '冠军',
            '亚军': '亚军', '第2名': '亚军', '第二名': '亚军', '2': '亚军', '2nd': '亚军',
            '季军': '第三名', '第3名': '第三名', '第三名': '第三名', '三名': '第三名', '3': '第三名', '3rd': '第三名',
            '第4名': '第四名', '第四名': '第四名', '四名': '第四名', '4': '第四名', '4th': '第四名',
            '第5名': '第五名', '第五名': '第五名', '五名': '第五名', '5': '第五名', '5th': '第五名',
            '第6名': '第六名', '第六名': '第六名', '六名': '第六名', '6': '第六名', '6th': '第六名',
            '第7名': '第七名', '第七名': '第七名', '七名': '第七名', '7': '第七名', '7th': '第七名',
            '第8名': '第八名', '第八名': '第八名', '八名': '第八名', '8': '第八名', '8th': '第八名',
            '第9名': '第九名', '第九名': '第九名', '九名': '第九名', '9': '第九名', '9th': '第九名',
            '第10名': '第十名', '第十名': '第十名', '十名': '第十名', '10': '第十名', '10th': '第十名'
        }
        
        position = position.strip()
        
        # 直接映射
        if position in position_mapping:
            return position_mapping[position]
        
        # 模糊匹配 - 增强逻辑
        for key, value in position_mapping.items():
            if key in position:
                return value
        
        # 处理带冒号的格式（如"第九名:"）
        if position.endswith(':'):
            clean_position = position[:-1].strip()
            if clean_position in position_mapping:
                return position_mapping[clean_position]
            for key, value in position_mapping.items():
                if key in clean_position:
                    return value
        
        # 如果还是无法识别，尝试更宽松的匹配
        position_lower = position.lower()
        if '九' in position_lower or '9' in position_lower:
            return '第九名'
        
        return position  # 返回原位置而不是未知

    def parse_3d_content(self, content):
        """解析3D投注内容 - 增强竖线格式支持"""
        content_str = str(content).strip()
        
        # 首先检查是否是竖线分隔格式
        if '|' in content_str and any(char.isdigit() or char == '_' or char == ',' for char in content_str):
            vertical_result = ContentParser.parse_3d_vertical_format(content_str)
            if any(vertical_result.values()):  # 如果有解析结果
                return vertical_result
        
        # 原有的解析逻辑
        return ContentParser.parse_positional_bets(content, ['百位', '十位', '个位'])
    
    def parse_lhc_special_content(self, content):
        """解析六合彩特殊玩法内容，按照玩法-投注内容格式解析"""
        content_str = str(content)
        
        # 新的解析逻辑：按照"玩法-投注内容"格式解析
        if '-' in content_str:
            parts = content_str.split('-', 1)  # 只分割第一个"-"
            play_method = parts[0].strip()      # 玩法部分
            bet_content = parts[1].strip()      # 投注内容部分
            
            # 调试信息 - 显示解析过程
            
            # 返回投注内容部分，这才是实际的下注内容
            return bet_content
        else:
            # 如果没有"-"，整个内容作为投注内容
            return content_str.strip()
    
    def extract_lhc_two_sides_content(self, content):
        """专门提取六合彩两面玩法的各种投注类型"""
        content_str = str(content)
        result = {
            'normal_size': set(),    # 普通大小：大/小
            'tail_size': set(),      # 尾大小：尾大/尾小
            'parity': set(),         # 单双：单/双
            'sum_parity': set(),     # 合数单双：合单/合双
            'range_bet': set(),      # 区间：1-10,11-20,21-30,31-40,41-49
            'animal_type': set(),    # 家禽野兽：家禽/野兽
            'zodiac': set(),         # 生肖
            'wave': set(),           # 波色：红波/蓝波/绿波
            'other': set()           # 其他
        }
    
        # 首先解析玩法-投注内容格式
        clean_content = content_str
        if '-' in content_str:
            parts = content_str.split('-', 1)
            clean_content = parts[1].strip()  # 只使用投注内容部分
    
        # 新增：特单、特双映射到普通单双
        if '特单' in clean_content:
            result['parity'].add('单')
        if '特双' in clean_content:
            result['parity'].add('双')
        
        # 新增：特家肖映射到家禽，特野肖映射到野兽
        if '特家肖' in clean_content or '家肖' in clean_content:
            result['animal_type'].add('家禽')
        if '特野肖' in clean_content or '野肖' in clean_content:
            result['animal_type'].add('野兽')
    
        # 波色检测
        if '红波' in clean_content and '红波-' not in content_str:
            result['wave'].add('红波')
        if '蓝波' in clean_content and '蓝波-' not in content_str:
            result['wave'].add('蓝波')
        if '绿波' in clean_content and '绿波-' not in content_str:
            result['wave'].add('绿波')
    
        # 普通大小检测
        if '大' in clean_content and '尾大' not in clean_content and '合大' not in clean_content and '特大' not in clean_content:
            result['normal_size'].add('大')
        if '小' in clean_content and '尾小' not in clean_content and '合小' not in clean_content and '特小' not in clean_content:
            result['normal_size'].add('小')
    
        # 尾大小检测
        if '尾大' in clean_content:
            result['tail_size'].add('尾大')
        if '尾小' in clean_content:
            result['tail_size'].add('尾小')
    
        # 单双检测（特单特双已经在上面处理了，这里处理普通单双）
        if '单' in clean_content and '合单' not in clean_content and '特单' not in clean_content:
            result['parity'].add('单')
        if '双' in clean_content and '合双' not in clean_content and '特双' not in clean_content:
            result['parity'].add('双')
    
        # 合数单双检测
        if '合单' in clean_content:
            result['sum_parity'].add('合单')
        if '合双' in clean_content:
            result['sum_parity'].add('合双')
    
        # 区间检测
        range_keywords = ['1-10', '11-20', '21-30', '31-40', '41-49']
        for range_keyword in range_keywords:
            if range_keyword in clean_content:
                result['range_bet'].add(range_keyword)
    
        # 家禽野兽检测（特家肖特野肖已经在上面处理了，这里处理普通的家禽野兽）
        if '家禽' in clean_content:
            result['animal_type'].add('家禽')
        if '野兽' in clean_content:
            result['animal_type'].add('野兽')
    
        # 生肖检测
        zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        for zodiac in zodiacs:
            if zodiac in clean_content:
                result['zodiac'].add(zodiac)
    
        # 清理空集合
        for key in list(result.keys()):
            if not result[key]:
                del result[key]
    
        return result

# ==================== 玩法分类统一 ====================
class PlayCategoryNormalizer:
    def __init__(self):
        self.category_mapping = self._create_category_mapping()
    
    def _create_category_mapping(self):
        """创建玩法分类映射的完整映射"""
        mapping = {
            # 快三玩法
            '和值': '和值',
            '和值_大小单双': '和值',
            '两面': '两面',
            '二不同号': '二不同号',
            '三不同号': '三不同号',
            # 注释掉同号玩法
            # '二同号': '二同号',
            # '三同号': '三同号',
            '独胆': '独胆',
            # 新增点数映射
            '点数': '和值',
            # 增强三军映射
            '三军': '独胆',
            '三軍': '独胆',
            '三军_大小': '独胆',
            '三军_单双': '独胆',
            
            # 六合彩玩法完整映射 - 尾数独立映射
            '特码': '特码',
            '正1特': '正1特',
            '正码特_正一特': '正1特',
            '正2特': '正2特',
            '正码特_正二特': '正2特',
            '正3特': '正3特',
            '正码特_正三特': '正3特',
            '正4特': '正4特',
            '正码特_正四特': '正4特',
            '正5特': '正5特',
            '正码特_正五特': '正5特',
            '正6特': '正6特',
            '正码特_正六特': '正6特',
            '正码': '正码',
            '正特': '正特',
            '正玛特': '正特',
            '正码1-6': '正码',
            
            # 尾数相关玩法独立映射
            '尾数': '尾数',
            '尾数_头尾数': '尾数_头尾数',  # 独立映射
            '特尾': '特尾',              # 独立映射
            '全尾': '全尾',              # 独立映射
            '尾数_正特尾数': '尾数',
            
            # 其他六合彩玩法
            '特肖': '特肖',
            '生肖_特肖': '特肖',
            '平特': '平特',
            '生肖_正肖': '平特',
            '生肖_一肖': '一肖',
            '连肖': '连肖',
            '连尾': '连尾',
            '龙虎': '龙虎',
            '五行': '五行',

            # 连肖玩法映射
            '二连肖': '二连肖',
            '三连肖': '三连肖', 
            '四连肖': '四连肖',
            '五连肖': '五连肖',
            '二连肖(中)': '二连肖',
            '三连肖(中)': '三连肖',
            '四连肖(中)': '四连肖', 
            '五连肖(中)': '五连肖',
            '连肖连尾_二连肖': '二连肖',
            '连肖连尾_三连肖': '三连肖',
            '连肖连尾_四连肖': '四连肖',
            '连肖连尾_五连肖': '五连肖',
            '连肖': '连肖',  # 通用连肖
            
            # 连尾玩法映射
            '二连尾': '二连尾',
            '三连尾': '三连尾',
            '四连尾': '四连尾',
            '五连尾': '五连尾',
            '连肖连尾_二连尾': '二连尾',
            '连肖连尾_三连尾': '三连尾',
            '连肖连尾_四连尾': '四连尾',
            '连肖连尾_五连尾': '五连尾',
            '连尾': '连尾',  # 通用连尾

            # 波色相关玩法
            '色波': '色波',
            '七色波': '色波',
            '波色': '色波',

            #半波相关玩法映射
            '半波': '半波',
            '蓝波': '半波',
            '绿波': '半波',
            '红波': '半波',
            '半波_红波': '半波',
            '半波_蓝波': '半波',
            '半波_绿波': '半波',

            # 正码1-6相关映射
            '正码1-6': '正码1-6',
            '正码1~6': '正码1-6',
            '正码1-6特': '正码1-6',
            '正码1~6特': '正码1-6',

            # 3D系列玩法映射
            '两面': '两面',
            '大小单双': '两面',
            '百位': '百位',
            '十位': '十位', 
            '个位': '个位',
            '百十': '百十',
            '百个': '百个',
            '十个': '十个',
            '百十个': '百十个',
            '定位胆': '定位胆',
            '定位胆_百位': '定位胆_百位',
            '定位胆_十位': '定位胆_十位',
            '定位胆_个位': '定位胆_个位',
            '百位(定位)': '定位胆_百位',
            '十位(定位)': '定位胆_十位',
            '个位(定位)': '定位胆_个位',
            
            # 时时彩玩法
            '斗牛': '斗牛',
            '1-5球': '1-5球',
            '第1球': '第1球',
            '第2球': '第2球',
            '第3球': '第3球',
            '第4球': '第4球',
            '第5球': '第5球',
            '总和': '总和',
            '正码': '正码',
            '正码特': '正码',
            '正码_特': '正码',
            '定位胆': '定位胆',
            '定位_万位': '定位_万位',
            '定位_千位': '定位_千位',
            '定位_百位': '定位_百位',
            '定位_十位': '定位_十位',
            '定位_个位': '定位_个位',
            '两面': '两面',
            
            # PK拾/赛车玩法
            '前一': '冠军',  # 前一就是冠军
            '定位胆': '定位胆',
            '1-5名': '1-5名',
            '6-10名': '6-10名',
            '冠军': '冠军',
            '亚军': '亚军',
            '季军': '第三名',
            '第3名': '第三名',
            '第4名': '第四名',
            '第5名': '第五名',
            '第6名': '第六名',
            '第7名': '第七名',
            '第8名': '第八名',
            '第9名': '第九名',
            '第10名': '第十名',
            '双面': '两面',
            '冠亚和': '冠亚和',
            '冠亚和_大小单双': '冠亚和_大小单双',
            '冠亚和_和值': '冠亚和_和值',
            
            # 大小单双独立玩法
            '大小_冠军': '大小_冠军',
            '大小_亚军': '大小_亚军',
            '大小_季军': '大小_季军',
            '单双_冠军': '单双_冠军',
            '单双_亚军': '单双_亚军',
            '单双_季军': '单双_季军',
            
            # 龙虎独立玩法
            '龙虎_冠军': '龙虎_冠军',
            '龙虎_冠 军': '龙虎_冠军',
            '龙虎_亚军': '龙虎_亚军',
            '龙虎_亚 军': '龙虎_亚军',
            '龙虎_季军': '龙虎_季军',
            '龙虎_季 军': '龙虎_季军',
            
            # 定位胆细分
            '定位胆_第1~5名': '定位胆_第1~5名',
            '定位胆_第6~10名': '定位胆_第6~10名',
            '定位胆_1~5': '定位胆_第1~5名',
            '定位胆_6~10': '定位胆_第6~10名',
            '定位胆_1-5': '定位胆_第1~5名', 
            '定位胆_6-10': '定位胆_第6~10名',
            '定位胆_1~5名': '定位胆_第1~5名',
            '定位胆_6~10名': '定位胆_第6~10名',
            
            # 大小单双玩法变体
            '大小单双': '两面',
            '大小': '大小',
            '单双': '单双',
            
            # 龙虎玩法变体
            '龙虎斗': '龙虎',
            '冠亚龙虎': '龙虎_冠军',
            '冠军龙虎': '龙虎_冠军',
            
            # 时时彩定位胆变体
            '定位_万位': '定位_万位',
            '定位_千位': '定位_千位', 
            '定位_百位': '定位_百位',
            '定位_十位': '定位_十位',
            '定位_个位': '定位_个位',
            '万位': '定位_万位',
            '千位': '定位_千位',
            '百位': '定位_百位',
            '十位': '定位_十位',
            '个位': '定位_个位',
            
            # 六合彩玩法变体
            '特码A': '特码',
            '特码B': '特码', 
            '正码A': '正码',
            '正码B': '正码',
            '正码1': '正1特',
            '正码2': '正2特',
            '正码3': '正3特',
            '正码4': '正4特',
            '正码5': '正5特',
            '正码6': '正6特',
            
            # 三色彩
            '正码': '正码',
            '两面': '两面',
            '色波': '色波',
            '特码': '特码'
        }
        return mapping
    
    def normalize_category(self, category):
        """统一玩法分类名称"""
        category_str = str(category).strip()
        
        # 直接映射
        if category_str in self.category_mapping:
            return self.category_mapping[category_str]
        
        # 关键词匹配
        for key, value in self.category_mapping.items():
            if key in category_str:
                return value
        
        category_lower = category_str.lower()
        
        # PK10/赛车智能匹配 - 补充更多变体
        if any(word in category_lower for word in ['定位胆_第1~5名', '定位胆1~5', '定位胆1-5']):
            return '定位胆_第1~5名'
        elif any(word in category_lower for word in ['定位胆_第6~10名', '定位胆6~10', '定位胆6-10']):
            return '定位胆_第6~10名'
        elif any(word in category_lower for word in ['1-5名', '1~5名', '1-5', '1~5']):
            return '1-5名'
        elif any(word in category_lower for word in ['6-10名', '6~10名', '6-10', '6~10']):
            return '6-10名'
        elif any(word in category_lower for word in ['冠军', '第一名', '第1名', '1st']):
            return '冠军'
        elif any(word in category_lower for word in ['亚军', '第二名', '第2名', '2nd']):
            return '亚军'
        elif any(word in category_lower for word in ['第三名', '第3名', '季军', '3rd']):
            return '第三名'
        elif any(word in category_lower for word in ['第四名', '第4名', '4th']):
            return '第四名'
        elif any(word in category_lower for word in ['第五名', '第5名', '5th']):
            return '第五名'
        elif any(word in category_lower for word in ['第六名', '第6名', '6th']):
            return '第六名'
        elif any(word in category_lower for word in ['第七名', '第7名', '7th']):
            return '第七名'
        elif any(word in category_lower for word in ['第八名', '第8名', '8th']):
            return '第八名'
        elif any(word in category_lower for word in ['第九名', '第9名', '9th']):
            return '第九名'
        elif any(word in category_lower for word in ['第十名', '第10名', '10th']):
            return '第十名'
        elif any(word in category_lower for word in ['前一']):
            return '冠军'  # 前一就是冠军
        
        # 时时彩定位胆智能匹配
        elif any(word in category_lower for word in ['万位', '第一位', '第一球']):
            return '定位_万位'
        elif any(word in category_lower for word in ['千位', '第二位', '第二球']):
            return '定位_千位'
        elif any(word in category_lower for word in ['百位', '第三位', '第三球']):
            return '定位_百位'
        elif any(word in category_lower for word in ['十位', '第四位', '第四球']):
            return '定位_十位'
        elif any(word in category_lower for word in ['个位', '第五位', '第五球']):
            return '定位_个位'
        elif any(word in category_lower for word in ['定位胆']):
            return '定位胆'
        
        # 六合彩智能匹配
        elif any(word in category_lower for word in ['特码']):
            return '特码'
        elif any(word in category_lower for word in ['正码']):
            return '正码'
        elif any(word in category_lower for word in ['正特', '正玛特']):
            return '正特'
        elif any(word in category_lower for word in ['尾数']):
            return '尾数'
        elif any(word in category_lower for word in ['平特']):
            return '平特'
        elif any(word in category_lower for word in ['特肖']):
            return '特肖'
        elif any(word in category_lower for word in ['一肖']):
            return '一肖'
        elif any(word in category_lower for word in ['连肖']):
            return '连肖'
        elif any(word in category_lower for word in ['连尾']):
            return '连尾'
        elif any(word in category_lower for word in ['龙虎']):
            return '龙虎'
        elif any(word in category_lower for word in ['五行']):
            return '五行'
        elif any(word in category_lower for word in ['色波', '七色波', '波色']):  # 统一色波识别
            return '色波'
        elif any(word in category_lower for word in ['半波']):
            return '半波'
        
        # 快三智能匹配 - 增强三军识别
        elif any(word in category_lower for word in ['和值', '点数']):
            return '和值'
        elif any(word in category_lower for word in ['独胆', '三军', '三軍']):  # 增强三军识别
            return '独胆'
        elif any(word in category_lower for word in ['二不同号']):
            return '二不同号'
        elif any(word in category_lower for word in ['三不同号']):
            return '三不同号'
        
        return category_str

# ==================== 替换：统一分析器 ====================
class UnifiedAnalyzer:
    """统一分析器 - 处理所有彩种"""
    
    def __init__(self):
        self.content_parser = ContentParser()
        self.normalizer = PlayCategoryNormalizer()
        self.data_analyzer = DataAnalyzer()
        self.threshold_manager = PrecisionThresholdManager()
        self.seen_records = set()
    
    def analyze_all_patterns(self, df):
        """统一分析入口"""
        all_results = {}
        
        # 重置缓存
        self.seen_records = set()
        
        # 使用进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 按彩种类型分别分析
        lottery_types = ['LHC', 'K3', 'PK10', 'SSC', '3D', 'THREE_COLOR']
        
        for i, lottery_type in enumerate(lottery_types):
            status_text.text(f"正在分析 {lottery_type}...")
            
            df_target = self._filter_by_lottery_type(df, lottery_type)
            if len(df_target) > 0:
                lottery_results = self._analyze_lottery_type(df_target, lottery_type)
                all_results[lottery_type] = lottery_results
            
            progress_bar.progress((i + 1) / len(lottery_types))
        
        status_text.text("分析完成！")
        return all_results
    
    def _filter_by_lottery_type(self, df, lottery_type):
        """过滤指定彩种类型的数据"""
        return df[df['彩种'].apply(self._identify_lottery_type) == lottery_type]
    
    def _identify_lottery_type(self, lottery_name):
        """识别彩种类型 - 复用原有逻辑"""
        lottery_str = str(lottery_name).strip()
        
        for lottery_type, config in LOTTERY_CONFIGS.items():
            for lottery in config['lotteries']:
                if lottery in lottery_str:
                    return lottery_type
        
        lottery_lower = lottery_str.lower()
        
        if any(word in lottery_lower for word in ['pk', '飞艇', '赛车', '幸运10', 'pk10', 'pk拾', '赛車']):
            return 'PK10'
        elif any(word in lottery_lower for word in ['快三', '快3', 'k3', 'k三']):
            return 'K3'
        elif any(word in lottery_lower for word in ['六合', 'lhc', '六合彩', '⑥合', '6合']):
            return 'LHC'
        elif any(word in lottery_lower for word in ['时时彩', 'ssc', '分分彩', '时时彩', '時時彩']):
            return 'SSC'
        elif any(word in lottery_lower for word in ['三色', '三色彩', '三色球']):
            return 'THREE_COLOR'
        elif any(word in lottery_lower for word in ['排列三', '排列3', '福彩3d', '3d', '极速3d', '排列', 'p3', 'p三']):
            return '3D'
        
        return None
    
    def _analyze_lottery_type(self, df, lottery_type):
        """分析指定彩种类型"""
        results = defaultdict(list)
        
        # 按账户期号分组分析
        grouped = df.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 按玩法分类分别分析
            for play_category, category_group in group.groupby('玩法分类'):
                category_results = self._analyze_by_play_category(
                    account, lottery, period, category_group, play_category, lottery_type
                )
                for result_type, records in category_results.items():
                    results[result_type].extend(records)
        
        return results
    
    def _analyze_by_play_category(self, account, lottery, period, group, play_category, lottery_type):
        """按玩法分类分析"""
        # 路由到对应的分析方法
        analysis_routes = {
            'LHC': self._analyze_lhc_play_category,
            'K3': self._analyze_k3_play_category,
            'PK10': self._analyze_pk10_play_category,
            'SSC': self._analyze_ssc_play_category, 
            '3D': self._analyze_3d_play_category,
            'THREE_COLOR': self._analyze_three_color_play_category
        }
        
        if lottery_type in analysis_routes:
            return analysis_routes[lottery_type](account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    # ==================== 六合彩分析方法 ====================
    def _analyze_lhc_play_category(self, account, lottery, period, group, play_category):
        """六合彩玩法分类分析"""
        results = defaultdict(list)
        
        # 六合彩玩法路由
        lhc_analysis_methods = {
            '特码': self._analyze_lhc_tema,
            '正码': self._analyze_lhc_zhengma,
            '正1特': self._analyze_lhc_zhengte,
            '正2特': self._analyze_lhc_zhengte,
            '正3特': self._analyze_lhc_zhengte,
            '正4特': self._analyze_lhc_zhengte,
            '正5特': self._analyze_lhc_zhengte,
            '正6特': self._analyze_lhc_zhengte,
            '尾数': self._analyze_lhc_tail,
            '特尾': self._analyze_lhc_tail,
            '全尾': self._analyze_lhc_tail,
            '平特': self._analyze_lhc_zodiac,
            '特肖': self._analyze_lhc_zodiac,
            '一肖': self._analyze_lhc_zodiac,
            '两面': self._analyze_lhc_two_sides,
            '色波': self._analyze_lhc_wave,
            '半波': self._analyze_lhc_banbo,
            '五行': self._analyze_lhc_five_elements,
            '二连肖': self._analyze_lhc_lianxiao,
            '三连肖': self._analyze_lhc_lianxiao,
            '四连肖': self._analyze_lhc_lianxiao,
            '五连肖': self._analyze_lhc_lianxiao,
            '二连尾': self._analyze_lhc_lianwei,
            '三连尾': self._analyze_lhc_lianwei,
            '四连尾': self._analyze_lhc_lianwei,
            '五连尾': self._analyze_lhc_lianwei
        }
        
        if play_category in lhc_analysis_methods:
            method = lhc_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_lhc_tema(self, account, lottery, period, group, play_category):
        """六合彩特码分析"""
        return self._analyze_number_bets(account, lottery, period, group, play_category, 'LHC', '特码多码')
    
    def _analyze_lhc_zhengma(self, account, lottery, period, group, play_category):
        """六合彩正码分析"""
        return self._analyze_number_bets(account, lottery, period, group, play_category, 'LHC', '正码多码')
    
    def _analyze_lhc_zhengte(self, account, lottery, period, group, play_category):
        """六合彩正码特分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        all_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            clean_content = self._parse_lhc_content(content)
            
            numbers = self.data_analyzer.extract_numbers_from_content(clean_content, 1, 49)
            all_numbers.update(numbers)
            
            two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
            for bet_type, bets in two_sides_analysis.items():
                all_bets[bet_type].update(bets)
        
        # 多码检测
        threshold = self.threshold_manager.get_threshold('LHC', play_category, 'multi_number')
        if len(all_numbers) >= threshold:
            record = self._create_number_record(
                account, lottery, period, play_category, all_numbers, '正特多码'
            )
            self._add_unique_result(results, '正特多码', record)
        
        # 矛盾检测
        conflicts = []
        wave_set = all_bets.get('wave', set())
        
        if '大' in all_bets.get('normal_size', set()) and '小' in all_bets.get('normal_size', set()):
            conflicts.append('大小矛盾')
        if '单' in all_bets.get('parity', set()) and '双' in all_bets.get('parity', set()):
            conflicts.append('单双矛盾')
        if len(wave_set) >= THRESHOLD_CONFIG['LHC']['wave_bet']:
            conflicts.append('波色多组投注')
        
        if conflicts:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflicts),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '正特矛盾')
            }
            self._add_unique_result(results, '正特矛盾', record)
        
        return results
    
    def _analyze_lhc_tail(self, account, lottery, period, group, play_category):
        """六合彩尾数分析"""
        results = defaultdict(list)
        
        all_tails = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            clean_content = self._parse_lhc_content(content)
            tails = self.data_analyzer.extract_tails_from_content(clean_content)
            all_tails.update(tails)
        
        # 根据不同的尾数玩法使用不同的结果键
        if play_category == '尾数':
            result_key = '尾数多码'
        elif play_category == '特尾':
            result_key = '特尾多尾'
        elif play_category == '全尾':
            result_key = '全尾多尾'
        else:
            result_key = '尾数多码'
        
        threshold = self.threshold_manager.get_threshold('LHC', play_category, 'multi_tail')
        if len(all_tails) >= threshold:
            bet_content = ', '.join([f"{tail}尾" for tail in sorted(all_tails)])
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': f"{play_category}（{len(all_tails)}尾）",
                '尾数数量': len(all_tails),
                '投注内容': bet_content,
                '排序权重': self._calculate_sort_weight({'尾数数量': len(all_tails)}, result_key)
            }
            self._add_unique_result(results, result_key, record)
        
        return results
    
    def _analyze_lhc_zodiac(self, account, lottery, period, group, play_category):
        """六合彩生肖分析"""
        results = defaultdict(list)
        
        all_zodiacs = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            clean_content = self._parse_lhc_content(content)
            zodiacs = self.data_analyzer.extract_zodiacs_from_content(clean_content)
            all_zodiacs.update(zodiacs)
        
        threshold = self.threshold_manager.get_threshold('LHC', play_category, 'zodiac_play')
        if len(all_zodiacs) >= threshold:
            # 根据玩法分类确定结果键
            if play_category == '平特':
                result_key = '平特多肖'
            elif play_category == '特肖':
                result_key = '特肖多肖'
            elif play_category == '一肖':
                result_key = '一肖多肖'
            else:
                result_key = '生肖多肖'
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '生肖数量': len(all_zodiacs),
                '投注内容': ', '.join(sorted(all_zodiacs)),
                '排序权重': self._calculate_sort_weight({'生肖数量': len(all_zodiacs)}, result_key)
            }
            self._add_unique_result(results, result_key, record)
        
        return results
    
    def _analyze_lhc_two_sides(self, account, lottery, period, group, play_category):
        """六合彩两面玩法分析"""
        results = defaultdict(list)
        
        all_bets = {
            'range_bet': set(),
            'normal_size': set(),
            'tail_size': set(),
            'parity': set(),
            'sum_parity': set(),
            'animal_type': set(),
            'zodiac': set(),
            'wave': set()
        }
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
            
            for bet_type in two_sides_analysis:
                if bet_type in all_bets:
                    all_bets[bet_type].update(two_sides_analysis[bet_type])
        
        # 区间多组检测
        if len(all_bets['range_bet']) >= THRESHOLD_CONFIG['LHC']['range_bet']:
            sorted_ranges = sorted(list(all_bets['range_bet']))
            bet_content = ', '.join(sorted_ranges)
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '投注区间数': len(all_bets['range_bet']),
                '投注区间': sorted_ranges,
                '投注内容': bet_content,
                '排序权重': self._calculate_sort_weight({'投注区间数': len(all_bets['range_bet'])}, '区间多组')
            }
            self._add_unique_result(results, '区间多组', record)
        
        # 矛盾检测
        conflict_types = []
        
        if '大' in all_bets.get('normal_size', set()) and '小' in all_bets.get('normal_size', set()):
            conflict_types.append('大小矛盾')
        if '尾大' in all_bets.get('tail_size', set()) and '尾小' in all_bets.get('tail_size', set()):
            conflict_types.append('尾大小矛盾')
        if '单' in all_bets.get('parity', set()) and '双' in all_bets.get('parity', set()):
            conflict_types.append('单双矛盾')
        if '合单' in all_bets.get('sum_parity', set()) and '合双' in all_bets.get('sum_parity', set()):
            conflict_types.append('合数单双矛盾')
        if '家禽' in all_bets.get('animal_type', set()) and '野兽' in all_bets.get('animal_type', set()):
            conflict_types.append('家禽野兽矛盾')
        
        if conflict_types:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflict_types),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面玩法矛盾')
            }
            self._add_unique_result(results, '两面玩法矛盾', record)
        
        # 波色检测
        wave_set = all_bets.get('wave', set())
        if len(wave_set) >= THRESHOLD_CONFIG['LHC']['wave_bet']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '投注波色数': len(wave_set),
                '投注波色': sorted(list(wave_set)),
                '排序权重': self._calculate_sort_weight({'投注波色数': len(wave_set)}, '波色三组')
            }
            self._add_unique_result(results, '波色三组', record)
        
        return results

    def _analyze_lhc_wave(self, account, lottery, period, group, play_category):
        """六合彩波色精确分析"""
        results = defaultdict(list)
        
        all_wave_bets = set()
        all_banbo_bets = set()
        
        # 定义半波投注项
        banbo_items = {
            '红大', '红小', '红单', '红双',
            '蓝大', '蓝小', '蓝单', '蓝双', 
            '绿大', '绿小', '绿单', '绿双'
        }
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            clean_content = self._parse_lhc_content(content)
            
            # 提取传统波色
            waves = self.data_analyzer.extract_wave_color_from_content(clean_content)
            all_wave_bets.update(waves)
            
            # 提取半波投注项
            for item in banbo_items:
                if item in clean_content:
                    all_banbo_bets.add(item)
        
        # 传统色波全包检测
        traditional_waves = {'红波', '蓝波', '绿波'}
        if traditional_waves.issubset(all_wave_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '违规类型': '色波全包',
                '投注波色数': len(traditional_waves),
                '投注波色': sorted(list(traditional_waves)),
                '投注内容': f"色波全包: {', '.join(sorted(traditional_waves))}",
                '排序权重': self._calculate_sort_weight({'投注波色数': len(traditional_waves)}, '色波全包')
            }
            self._add_unique_result(results, '色波全包', record)
        
        return results
    
    def _analyze_lhc_banbo(self, account, lottery, period, group, play_category):
        """六合彩半波精确分析"""
        results = defaultdict(list)
        
        all_banbo_bets = set()
        
        # 定义半波投注项
        size_full_set = {'红大', '红小', '蓝大', '蓝小', '绿大', '绿小'}
        parity_full_set = {'红单', '红双', '蓝单', '蓝双', '绿单', '绿双'}
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 解析玩法-投注内容格式
            if '-' in content:
                parts = content.split('-', 1)
                bet_content = parts[1].strip()
            else:
                bet_content = content
            
            # 提取所有半波投注项
            for bet in size_full_set.union(parity_full_set):
                if bet in bet_content:
                    all_banbo_bets.add(bet)
        
        # 大小全包检测
        if size_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '违规类型': '半波大小全包',
                '投注半波数': len(size_full_set),
                '投注半波': sorted(list(size_full_set)),
                '投注内容': ', '.join(sorted(size_full_set)),
                '排序权重': self._calculate_sort_weight({'投注半波数': len(size_full_set)}, '半波大小全包')
            }
            self._add_unique_result(results, '半波大小全包', record)
        
        # 单双全包检测
        if parity_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '违规类型': '半波单双全包',
                '投注半波数': len(parity_full_set),
                '投注半波': sorted(list(parity_full_set)),
                '投注内容': ', '.join(sorted(parity_full_set)),
                '排序权重': self._calculate_sort_weight({'投注半波数': len(parity_full_set)}, '半波单双全包')
            }
            self._add_unique_result(results, '半波单双全包', record)
        
        return results
    
    def _analyze_lhc_five_elements(self, account, lottery, period, group, play_category):
        """六合彩五行精确分析"""
        results = defaultdict(list)
        
        all_elements = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            clean_content = self._parse_lhc_content(content)
            elements = self.data_analyzer.extract_five_elements_from_content(clean_content)
            all_elements.update(elements)
        
        if len(all_elements) >= THRESHOLD_CONFIG['LHC']['five_elements']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '投注五行数': len(all_elements),
                '投注五行': sorted(list(all_elements)),
                '排序权重': self._calculate_sort_weight({'投注五行数': len(all_elements)}, '五行多组')
            }
            self._add_unique_result(results, '五行多组', record)
        
        return results
    
    def _analyze_lhc_lianxiao(self, account, lottery, period, group, play_category):
        """六合彩连肖精确分析"""
        results = defaultdict(list)
        
        all_zodiacs = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 解析玩法-投注内容格式
            if '-' in content:
                parts = content.split('-', 1)
                bet_content = parts[1].strip()
            else:
                bet_content = content
                
            zodiacs = self.data_analyzer.extract_zodiacs_from_content(bet_content)
            all_zodiacs.update(zodiacs)
        
        # 连肖类型阈值配置
        lianxiao_thresholds = {
            '二连肖': 7,
            '三连肖': 7,
            '四连肖': 7,
            '五连肖': 8
        }
        
        threshold = lianxiao_thresholds.get(play_category, 6)
        
        if len(all_zodiacs) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': f"{play_category}（{len(all_zodiacs)}生肖）",
                '违规类型': f'{play_category}多肖',
                '生肖数量': len(all_zodiacs),
                '投注内容': ', '.join(sorted(all_zodiacs)),
                '排序权重': self._calculate_sort_weight({'生肖数量': len(all_zodiacs)}, f'{play_category}多肖')
            }
            self._add_unique_result(results, f'{play_category}多肖', record)
        
        return results
    
    def _analyze_lhc_lianwei(self, account, lottery, period, group, play_category):
        """六合彩连尾精确分析"""
        results = defaultdict(list)
        
        all_tails = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            tails = self.data_analyzer.extract_tails_from_content(content)
            all_tails.update(tails)
        
        # 连尾类型阈值配置
        lianwei_thresholds = {
            '二连尾': 7,
            '三连尾': 7,
            '四连尾': 7,
            '五连尾': 8
        }
        
        threshold = lianwei_thresholds.get(play_category, 6)
        
        if len(all_tails) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': f"{play_category}（{len(all_tails)}尾）",
                '违规类型': f'{play_category}多尾',
                '尾数数量': len(all_tails),
                '投注内容': ', '.join([f"{tail}尾" for tail in sorted(all_tails)]),
                '排序权重': self._calculate_sort_weight({'尾数数量': len(all_tails)}, f'{play_category}多尾')
            }
            self._add_unique_result(results, f'{play_category}多尾', record)
        
        return results

    # ==================== 快三分析方法 ====================
    def _analyze_k3_play_category(self, account, lottery, period, group, play_category):
        """快三玩法分类分析"""
        results = defaultdict(list)
        
        # 快三玩法路由
        k3_analysis_methods = {
            '和值': self._analyze_k3_hezhi,
            '和值_大小单双': self._analyze_k3_hezhi,
            '独胆': self._analyze_k3_dudan,
            '二不同号': self._analyze_k3_different,
            '三不同号': self._analyze_k3_different,
            '两面': self._analyze_k3_two_sides
        }
        
        if play_category in k3_analysis_methods:
            method = k3_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_k3_hezhi(self, account, lottery, period, group, play_category):
        """快三和值分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        all_contents = []
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 提取数字
            numbers = self.data_analyzer.extract_numbers_from_content(
                content,
                LOTTERY_CONFIGS['K3']['hezhi_min'],
                LOTTERY_CONFIGS['K3']['hezhi_max']
            )
            all_numbers.update(numbers)
            all_contents.append(content)
            
            # 检查大小单双
            content_lower = content.lower()
            if '大' in content_lower:
                has_big = True
            if '小' in content_lower:
                has_small = True
            if '单' in content_lower:
                has_single = True
            if '双' in content_lower:
                has_double = True
        
        # 和值多码检测
        threshold = self.threshold_manager.get_threshold('K3', play_category, 'hezhi_multi_number')
        if len(all_numbers) >= threshold:
            bet_content = ', '.join([str(num) for num in sorted(all_numbers)])
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': bet_content,
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '和值多码')
            }
            self._add_unique_result(results, '和值多码', record)
            return results  # 避免重复记录
        
        # 和值矛盾检测
        conflict_types = []
        if has_big and has_small:
            conflict_types.append('大小')
        if has_single and has_double:
            conflict_types.append('单双')
        
        if conflict_types:
            bet_content_parts = []
            if has_big:
                bet_content_parts.append('大')
            if has_small:
                bet_content_parts.append('小')
            if has_single:
                bet_content_parts.append('单')
            if has_double:
                bet_content_parts.append('双')
            bet_content = ', '.join(bet_content_parts)
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflict_types),
                '投注内容': bet_content,
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '和值矛盾')
            }
            self._add_unique_result(results, '和值矛盾', record)
        
        # 和值大小矛盾检测
        if all_numbers and len(all_numbers) < threshold:
            small_values = [num for num in all_numbers if 3 <= num <= 10]
            big_values = [num for num in all_numbers if 11 <= num <= 18]
            single_values = [num for num in all_numbers if num % 2 == 1]
            double_values = [num for num in all_numbers if num % 2 == 0]
            
            # 收集所有可能的矛盾
            possible_contradictions = []
            
            if has_small and len(big_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(big_values)
                description = f"投注小但包含多个大号码(小{len(small_values)}个,大{len(big_values)}个)"
                possible_contradictions.append(('大小矛盾', description, contradiction_value))
            
            if has_big and len(small_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(small_values)
                description = f"投注大但包含多个小号码(小{len(small_values)}个,大{len(big_values)}个)"
                possible_contradictions.append(('大小矛盾', description, contradiction_value))
            
            if has_single and len(double_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(double_values)
                description = f"投注单但包含多个双号码(单{len(single_values)}个,双{len(double_values)}个)"
                possible_contradictions.append(('单双矛盾', description, contradiction_value))
            
            if has_double and len(single_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(single_values)
                description = f"投注双但包含多个单号码(单{len(single_values)}个,双{len(double_values)}个)"
                possible_contradictions.append(('单双矛盾', description, contradiction_value))
            
            if possible_contradictions:
                possible_contradictions.sort(key=lambda x: x[2], reverse=True)
                best_contradiction = possible_contradictions[0]
                contradiction_type, contradiction_desc, contradiction_value = best_contradiction
                
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '矛盾类型': contradiction_desc,
                    '矛盾值': contradiction_value,
                    '大号码数量': len(big_values),
                    '小号码数量': len(small_values),
                    '单号码数量': len(single_values),
                    '双号码数量': len(double_values),
                    '排序权重': self._calculate_sort_weight({'矛盾值': contradiction_value}, '和值大小矛盾')
                }
                self._add_unique_result(results, '和值大小矛盾', record)
        
        return results
    
    def _analyze_k3_dudan(self, account, lottery, period, group, play_category):
        """快三独胆分析"""
        results = defaultdict(list)
        
        # 聚合同一账户同一期号的所有独胆投注
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 6)
            all_numbers.update(numbers)
        
        threshold = self.threshold_manager.get_threshold('K3', play_category, 'dudan_multi_number')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"聚合投注: {', '.join([str(num) for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '独胆多码')
            }
            self._add_unique_result(results, '独胆多码', record)
        
        return results
    
    def _analyze_k3_different(self, account, lottery, period, group, play_category):
        """快三不同号分析"""
        results = defaultdict(list)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 6)
            
            if len(numbers) == 6:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '号码数量': len(numbers),
                    '投注内容': ', '.join([str(num) for num in sorted(numbers)]),
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '不同号全包')
                }
                self._add_unique_result(results, '不同号全包', record)
        
        return results
    
    def _analyze_k3_two_sides(self, account, lottery, period, group, play_category):
        """快三两面分析"""
        results = defaultdict(list)
        
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            content_lower = content.lower()
            
            if '大' in content_lower:
                has_big = True
            if '小' in content_lower:
                has_small = True
            if '单' in content_lower:
                has_single = True
            if '双' in content_lower:
                has_double = True
        
        conflict_types = []
        if has_big and has_small:
            conflict_types.append('大小')
        if has_single and has_double:
            conflict_types.append('单双')
        
        if conflict_types:
            bet_content_parts = []
            if has_big:
                bet_content_parts.append('大')
            if has_small:
                bet_content_parts.append('小')
            if has_single:
                bet_content_parts.append('单')
            if has_double:
                bet_content_parts.append('双')
            bet_content = ', '.join(bet_content_parts)
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflict_types),
                '投注内容': bet_content,
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面矛盾')
            }
            self._add_unique_result(results, '两面矛盾', record)
        
        return results

    # ==================== PK10/赛车分析方法 ====================
    def _analyze_pk10_play_category(self, account, lottery, period, group, play_category):
        """PK10玩法分类分析"""
        results = defaultdict(list)
        
        # PK10玩法路由
        pk10_analysis_methods = {
            '冠军': self._analyze_pk10_position,
            '亚军': self._analyze_pk10_position,
            '第三名': self._analyze_pk10_position,
            '第四名': self._analyze_pk10_position,
            '第五名': self._analyze_pk10_position,
            '第六名': self._analyze_pk10_position,
            '第七名': self._analyze_pk10_position,
            '第八名': self._analyze_pk10_position,
            '第九名': self._analyze_pk10_position,
            '第十名': self._analyze_pk10_position,
            '前一': self._analyze_pk10_position,
            '1-5名': self._analyze_pk10_range,
            '6-10名': self._analyze_pk10_range,
            '定位胆': self._analyze_pk10_dingwei,
            '定位胆_第1~5名': self._analyze_pk10_dingwei,
            '定位胆_第6~10名': self._analyze_pk10_dingwei,
            '冠亚和': self._analyze_pk10_gyh,
            '冠亚和_大小单双': self._analyze_pk10_gyh,
            '冠亚和_和值': self._analyze_pk10_gyh,
            '两面': self._analyze_pk10_two_sides,
            '双面': self._analyze_pk10_two_sides,
            '大小_冠军': self._analyze_pk10_independent,
            '大小_亚军': self._analyze_pk10_independent,
            '大小_季军': self._analyze_pk10_independent,
            '单双_冠军': self._analyze_pk10_independent,
            '单双_亚军': self._analyze_pk10_independent,
            '单双_季军': self._analyze_pk10_independent,
            '龙虎_冠军': self._analyze_pk10_dragon_tiger,
            '龙虎_亚军': self._analyze_pk10_dragon_tiger,
            '龙虎_季军': self._analyze_pk10_dragon_tiger
        }
        
        if play_category in pk10_analysis_methods:
            method = pk10_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_pk10_position(self, account, lottery, period, group, play_category):
        """PK10位置号码分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析PK10内容
            bets_by_position = ContentParser.parse_pk10_content(content)
            
            for position, numbers in bets_by_position.items():
                # 如果解析出的位置匹配当前玩法分类，则收集号码
                if self._is_position_match(position, play_category):
                    all_numbers.update(numbers)
        
        # 多码检测
        threshold = self.threshold_manager.get_threshold('PK10', play_category, 'multi_number')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '位置': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"{play_category}-{','.join([f'{num:02d}' for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '超码')
            }
            self._add_unique_result(results, '超码', record)
        
        return results
    
    def _is_position_match(self, position, play_category):
        """检查位置是否匹配玩法分类"""
        position_mapping = {
            '冠军': ['冠军', '前一'],
            '亚军': ['亚军'],
            '第三名': ['第三名', '季军'],
            '第四名': ['第四名'],
            '第五名': ['第五名'],
            '第六名': ['第六名'],
            '第七名': ['第七名'],
            '第八名': ['第八名'],
            '第九名': ['第九名'],
            '第十名': ['第十名']
        }
        
        for key, values in position_mapping.items():
            if play_category == key and position in values:
                return True
        
        return play_category == position
    
    def _analyze_pk10_range(self, account, lottery, period, group, play_category):
        """PK10范围分析（1-5名、6-10名）"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 10, is_pk10=True)
            all_numbers.update(numbers)
        
        threshold = self.threshold_manager.get_threshold('PK10', play_category, 'multi_number')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"{play_category}: {', '.join([f'{num:02d}' for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '超码')
            }
            self._add_unique_result(results, '超码', record)
        
        return results
    
    def _analyze_pk10_dingwei(self, account, lottery, period, group, play_category):
        """PK10定位胆分析"""
        results = defaultdict(list)
        
        position_numbers = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析定位胆内容
            bets_by_position = ContentParser.parse_pk10_content(content)
            
            for position, numbers in bets_by_position.items():
                position_numbers[position].update(numbers)
        
        # 检查每个位置的超码
        for position, numbers in position_numbers.items():
            threshold = self.threshold_manager.get_threshold('PK10', '定位胆', 'multi_number')
            if len(numbers) >= threshold:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([f'{num:02d}' for num in sorted(numbers)])}",
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '超码')
                }
                self._add_unique_result(results, '超码', record)
        
        return results
    
    def _analyze_pk10_gyh(self, account, lottery, period, group, play_category):
        """PK10冠亚和分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        all_size_parity = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 提取冠亚和号码
            numbers = re.findall(r'\b\d{1,2}\b', content)
            numbers = [int(num) for num in numbers if 1 <= int(num) <= 19]
            all_numbers.update(numbers)
            
            # 提取大小单双
            size_parity = self.data_analyzer.extract_size_parity_from_content(content)
            all_size_parity.update(size_parity)
        
        # 冠亚和多码检测
        threshold = self.threshold_manager.get_threshold('PK10', '冠亚和', 'gyh_multi_number')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([str(num) for num in sorted(all_numbers)]),
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '冠亚和多码')
            }
            self._add_unique_result(results, '冠亚和多码', record)
            return results
        
        # 冠亚和矛盾检测
        conflicts = []
        if '大' in all_size_parity and '小' in all_size_parity:
            conflicts.append('大小')
        if '单' in all_size_parity and '双' in all_size_parity:
            conflicts.append('单双')
        
        if conflicts:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflicts),
                '投注内容': ', '.join(sorted(all_size_parity)),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '冠亚和矛盾')
            }
            self._add_unique_result(results, '冠亚和矛盾', record)
        
        return results
    
    def _analyze_pk10_two_sides(self, account, lottery, period, group, play_category):
        """PK10两面玩法分析"""
        results = defaultdict(list)
        
        position_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            if '-' in content:
                parts = content.split(',')
                for part in parts:
                    if '-' in part:
                        try:
                            position, bet_option = part.split('-', 1)
                            position = self.data_analyzer._normalize_pk10_position(position)
                            bet_option = bet_option.strip()
                            
                            if bet_option in ['大', '小', '单', '双', '龙', '虎']:
                                position_bets[position].add(bet_option)
                        except ValueError:
                            continue
        
        for position, bets in position_bets.items():
            conflicts = []
            
            if '大' in bets and '小' in bets:
                conflicts.append('大小')
            if '单' in bets and '双' in bets:
                conflicts.append('单双')
            if '龙' in bets and '虎' in bets:
                conflicts.append('龙虎')
            
            if conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
        
        return results
    
    def _analyze_pk10_independent(self, account, lottery, period, group, play_category):
        """PK10独立玩法分析（大小单双）"""
        results = defaultdict(list)
        
        position_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 确定位置
            if '冠军' in category or '前一' in category:
                position = '冠军'
            elif '亚军' in category:
                position = '亚军'
            elif '季军' in category:
                position = '第三名'
            else:
                continue
            
            if '大小' in category:
                bets = self.data_analyzer.extract_size_parity_from_content(content)
            elif '单双' in category:
                bets = self.data_analyzer.extract_size_parity_from_content(content)
            else:
                bets = []
            
            position_bets[position].update(bets)
        
        for position, bets in position_bets.items():
            conflicts = []
            
            if '大' in bets and '小' in bets:
                conflicts.append('大小')
            if '单' in bets and '双' in bets:
                conflicts.append('单双')
            
            if conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '独立玩法矛盾')
                }
                self._add_unique_result(results, '独立玩法矛盾', record)
        
        return results
    
    def _analyze_pk10_dragon_tiger(self, account, lottery, period, group, play_category):
        """PK10龙虎分析"""
        results = defaultdict(list)
        
        position_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 确定位置
            if '冠军' in category:
                position = '冠军'
            elif '亚军' in category:
                position = '亚军'
            elif '季军' in category:
                position = '第三名'
            else:
                # 从内容推断位置
                position = self.data_analyzer._infer_pk10_position_from_content(content)
            
            # 提取龙虎投注
            dragon_tiger = self.data_analyzer.extract_dragon_tiger_from_content(content)
            position_bets[position].update(dragon_tiger)
        
        # 检查矛盾
        for position, bets in position_bets.items():
            if '龙' in bets and '虎' in bets:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '矛盾类型': '龙虎矛盾',
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '龙虎矛盾'}, '龙虎矛盾')
                }
                self._add_unique_result(results, '龙虎矛盾', record)
        
        return results

    # ==================== 时时彩分析方法 ====================
    def _analyze_ssc_play_category(self, account, lottery, period, group, play_category):
        """时时彩玩法分类分析"""
        results = defaultdict(list)
        
        # 时时彩玩法路由
        ssc_analysis_methods = {
            '第1球': self._analyze_ssc_position,
            '第2球': self._analyze_ssc_position,
            '第3球': self._analyze_ssc_position,
            '第4球': self._analyze_ssc_position,
            '第5球': self._analyze_ssc_position,
            '万位': self._analyze_ssc_position,
            '千位': self._analyze_ssc_position,
            '百位': self._analyze_ssc_position,
            '十位': self._analyze_ssc_position,
            '个位': self._analyze_ssc_position,
            '1-5球': self._analyze_ssc_range,
            '定位胆': self._analyze_ssc_dingwei,
            '定位_万位': self._analyze_ssc_position,
            '定位_千位': self._analyze_ssc_position,
            '定位_百位': self._analyze_ssc_position,
            '定位_十位': self._analyze_ssc_position,
            '定位_个位': self._analyze_ssc_position,
            '两面': self._analyze_ssc_two_sides,
            '总和': self._analyze_ssc_zonghe,
            '斗牛': self._analyze_ssc_douniu
        }
        
        if play_category in ssc_analysis_methods:
            method = ssc_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_ssc_position(self, account, lottery, period, group, play_category):
        """时时彩位置分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析时时彩内容
            bets_by_position = ContentParser.parse_ssc_content(content)
            
            for position, numbers in bets_by_position.items():
                # 如果解析出的位置匹配当前玩法分类，则收集号码
                if self._is_ssc_position_match(position, play_category):
                    all_numbers.update(numbers)
        
        # 多码检测
        threshold = self.threshold_manager.get_threshold('SSC', play_category, 'dingwei_multi')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '位置': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"{play_category}-{','.join([str(num) for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '定位胆多码')
            }
            self._add_unique_result(results, '定位胆多码', record)
        
        return results
    
    def _is_ssc_position_match(self, position, play_category):
        """检查时时彩位置是否匹配玩法分类"""
        position_mapping = {
            '第1球': ['第1球', '万位', '定位_万位'],
            '第2球': ['第2球', '千位', '定位_千位'],
            '第3球': ['第3球', '百位', '定位_百位'],
            '第4球': ['第4球', '十位', '定位_十位'],
            '第5球': ['第5球', '个位', '定位_个位']
        }
        
        for key, values in position_mapping.items():
            if play_category == key and position in values:
                return True
        
        return play_category == position
    
    def _analyze_ssc_range(self, account, lottery, period, group, play_category):
        """时时彩范围分析（1-5球）"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 0, 9)
            all_numbers.update(numbers)
        
        threshold = self.threshold_manager.get_threshold('SSC', play_category, 'dingwei_multi')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"{play_category}: {', '.join([str(num) for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '定位胆多码')
            }
            self._add_unique_result(results, '定位胆多码', record)
        
        return results
    
    def _analyze_ssc_dingwei(self, account, lottery, period, group, play_category):
        """时时彩定位胆分析"""
        results = defaultdict(list)
        
        position_numbers = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析定位胆内容
            bets_by_position = ContentParser.parse_ssc_content(content)
            
            for position, numbers in bets_by_position.items():
                position_numbers[position].update(numbers)
        
        # 检查每个位置的超码
        for position, numbers in position_numbers.items():
            threshold = self.threshold_manager.get_threshold('SSC', '定位胆', 'dingwei_multi')
            if len(numbers) >= threshold:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([str(num) for num in sorted(numbers)])}",
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '定位胆多码')
                }
                self._add_unique_result(results, '定位胆多码', record)
        
        return results
    
    def _analyze_ssc_two_sides(self, account, lottery, period, group, play_category):
        """时时彩两面玩法分析"""
        results = defaultdict(list)
        
        total_bets = set()
        ball_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            if '总和、龙虎-' in content:
                clean_content = content.replace('总和、龙虎-', '')
                bets = clean_content.split(',')
                for bet in bets:
                    if '总和大' in bet:
                        total_bets.add('大')
                    elif '总和小' in bet:
                        total_bets.add('小')
                    elif '总和单' in bet:
                        total_bets.add('单')
                    elif '总和双' in bet:
                        total_bets.add('双')
                    elif '龙' in bet:
                        total_bets.add('龙')
                    elif '虎' in bet:
                        total_bets.add('虎')
            
            for i in range(1, 6):
                ball_key = f'第{i}球'
                if ball_key in content:
                    bets = self.data_analyzer.extract_size_parity_from_content(content)
                    ball_bets[ball_key].update(bets)
        
        conflicts = []
        if '大' in total_bets and '小' in total_bets:
            conflicts.append('总和大/小')
        if '单' in total_bets and '双' in total_bets:
            conflicts.append('总和单/双')
        if '龙' in total_bets and '虎' in total_bets:
            conflicts.append('龙/虎')
        
        if conflicts:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflicts),
                '投注内容': f"总和:{','.join(sorted(total_bets))}",
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '两面矛盾')
            }
            self._add_unique_result(results, '两面矛盾', record)
        
        for ball, bets in ball_bets.items():
            ball_conflicts = []
            if '大' in bets and '小' in bets:
                ball_conflicts.append('大小')
            if '单' in bets and '双' in bets:
                ball_conflicts.append('单双')
            
            if ball_conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '矛盾类型': f"{ball}{'、'.join(ball_conflicts)}",
                    '投注内容': f"{ball}:{','.join(sorted(bets))}",
                    '排序权重': self._calculate_sort_weight({'矛盾类型': f"{ball}{'、'.join(ball_conflicts)}"}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
        
        return results
    
    def _analyze_ssc_zonghe(self, account, lottery, period, group, play_category):
        """时时彩总和分析"""
        results = defaultdict(list)
        
        all_bets = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            bets = self.data_analyzer.extract_size_parity_from_content(content)
            all_bets.update(bets)
        
        conflicts = []
        if '大' in all_bets and '小' in all_bets:
            conflicts.append('大小')
        if '单' in all_bets and '双' in all_bets:
            conflicts.append('单双')
        
        if conflicts:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflicts),
                '投注内容': ', '.join(sorted(all_bets)),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '总和矛盾')
            }
            self._add_unique_result(results, '总和矛盾', record)
        
        return results
    
    def _analyze_ssc_douniu(self, account, lottery, period, group, play_category):
        """时时彩斗牛分析"""
        results = defaultdict(list)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            bull_types = self.data_analyzer.extract_douniu_types(content)
            
            if len(bull_types) >= THRESHOLD_CONFIG['SSC']['douniu_multi']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '号码数量': len(bull_types),
                    '投注内容': ', '.join(sorted(bull_types)),
                    '排序权重': self._calculate_sort_weight({'号码数量': len(bull_types)}, '斗牛多码')
                }
                self._add_unique_result(results, '斗牛多码', record)
        
        return results

    # ==================== 3D分析方法 ====================
    def _analyze_3d_play_category(self, account, lottery, period, group, play_category):
        """3D玩法分类分析"""
        results = defaultdict(list)
        
        # 3D玩法路由
        three_d_analysis_methods = {
            '两面': self._analyze_3d_two_sides,
            '大小单双': self._analyze_3d_two_sides,
            '定位胆': self._analyze_3d_dingwei,
            '定位胆_百位': self._analyze_3d_position,
            '定位胆_十位': self._analyze_3d_position,
            '定位胆_个位': self._analyze_3d_position,
            '百位': self._analyze_3d_position,
            '十位': self._analyze_3d_position,
            '个位': self._analyze_3d_position
        }
        
        if play_category in three_d_analysis_methods:
            method = three_d_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_3d_two_sides(self, account, lottery, period, group, play_category):
        """3D两面玩法分析"""
        results = defaultdict(list)
        
        # 按位置分类收集投注
        position_bets = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 首先尝试解析竖线格式
            bets_by_position = self.data_analyzer.parse_3d_content(content)
            if bets_by_position:
                # 竖线格式解析成功
                for position, bets in bets_by_position.items():
                    for bet in bets:
                        # 提取大小单双信息
                        if isinstance(bet, str):
                            if '大' in bet:
                                position_bets[position].add('大')
                            if '小' in bet:
                                position_bets[position].add('小')
                            if '单' in bet:
                                position_bets[position].add('单')
                            if '双' in bet:
                                position_bets[position].add('双')
            else:
                # 原有的解析逻辑
                positions = ['百位', '十位', '个位', '百十', '百个', '十个', '百十个']
                bets = ['大', '小', '单', '双', '质', '合', '和大', '和小', '和单', '和双', 
                       '和尾大', '和尾小', '和尾质', '和尾合']
                
                # 处理多种格式
                parts = [part.strip() for part in content.split(',')]
                
                current_position = None
                
                for part in parts:
                    # 检查是否包含位置信息
                    position_found = False
                    for position in positions:
                        if position in part:
                            current_position = position
                            position_found = True
                            break
                    
                    if position_found:
                        # 提取该位置的所有投注选项
                        for bet in bets:
                            if bet in part:
                                position_bets[current_position].add(bet)
                    elif current_position:
                        # 如果没有位置信息但有当前上下文位置，检查投注选项
                        for bet in bets:
                            if bet in part:
                                position_bets[current_position].add(bet)
        
        # 检查每个位置的矛盾
        for position, bet_options in position_bets.items():
            conflicts = []
            
            # 基本大小单双质合矛盾
            if '大' in bet_options and '小' in bet_options:
                conflicts.append('大小矛盾')
            if '单' in bet_options and '双' in bet_options:
                conflicts.append('单双矛盾')
            if '质' in bet_options and '合' in bet_options:
                conflicts.append('质合矛盾')
            
            # 和数属性矛盾
            if '和大' in bet_options and '和小' in bet_options:
                conflicts.append('和大小矛盾')
            if '和单' in bet_options and '和双' in bet_options:
                conflicts.append('和单双矛盾')
            if '和尾大' in bet_options and '和尾小' in bet_options:
                conflicts.append('和尾大小矛盾')
            if '和尾质' in bet_options and '和尾合' in bet_options:
                conflicts.append('和尾质合矛盾')
            
            if conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}:{','.join(sorted(bet_options))}",
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
        
        return results
    
    def _analyze_3d_dingwei(self, account, lottery, period, group, play_category):
        """3D定位胆分析"""
        results = defaultdict(list)
        
        position_numbers = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析3D内容
            bets_by_position = self.data_analyzer.parse_3d_content(content)
            if bets_by_position:
                # 如果有解析结果，使用解析出的位置和号码
                for position, numbers in bets_by_position.items():
                    position_numbers[position].update(numbers)
                continue
            
            # 如果没有竖线格式，使用原有逻辑
            # 确定位置
            if '百位' in content:
                position = '百位'
            elif '十位' in content:
                position = '十位'
            elif '个位' in content:
                position = '个位'
            else:
                position = '未知位置'
            
            # 提取号码
            numbers = self.data_analyzer.extract_numbers_from_content(content, 0, 9)
            position_numbers[position].update(numbers)
        
        # 检查每个位置的超码
        for position, numbers in position_numbers.items():
            threshold = self.threshold_manager.get_threshold('3D', '定位胆', 'dingwei_multi')
            if len(numbers) >= threshold:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([str(num) for num in sorted(numbers)])}",
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '定位胆多码')
                }
                self._add_unique_result(results, '定位胆多码', record)
        
        return results
    
    def _analyze_3d_position(self, account, lottery, period, group, play_category):
        """3D位置分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析3D内容
            bets_by_position = self.data_analyzer.parse_3d_content(content)
            
            for position, numbers in bets_by_position.items():
                # 如果解析出的位置匹配当前玩法分类，则收集号码
                if self._is_3d_position_match(position, play_category):
                    all_numbers.update(numbers)
        
        # 多码检测
        threshold = self.threshold_manager.get_threshold('3D', play_category, 'dingwei_multi')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '位置': play_category,
                '号码数量': len(all_numbers),
                '投注内容': f"{play_category}-{','.join([str(num) for num in sorted(all_numbers)])}",
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '定位胆多码')
            }
            self._add_unique_result(results, '定位胆多码', record)
        
        return results
    
    def _is_3d_position_match(self, position, play_category):
        """检查3D位置是否匹配玩法分类"""
        position_mapping = {
            '定位胆_百位': ['百位'],
            '定位胆_十位': ['十位'],
            '定位胆_个位': ['个位']
        }
        
        for key, values in position_mapping.items():
            if play_category == key and position in values:
                return True
        
        return play_category == position

    # ==================== 三色彩分析方法 ====================
    def _analyze_three_color_play_category(self, account, lottery, period, group, play_category):
        """三色彩玩法分类分析"""
        results = defaultdict(list)
        
        # 三色彩玩法路由
        three_color_analysis_methods = {
            '正码': self._analyze_three_color_zhengma,
            '两面': self._analyze_three_color_two_sides,
            '色波': self._analyze_three_color_wave,
            '特码': self._analyze_three_color_tema
        }
        
        if play_category in three_color_analysis_methods:
            method = three_color_analysis_methods[play_category]
            return method(account, lottery, period, group, play_category)
        
        return defaultdict(list)
    
    def _analyze_three_color_zhengma(self, account, lottery, period, group, play_category):
        """三色彩正码分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 0, 9)
            all_numbers.update(numbers)
        
        threshold = self.threshold_manager.get_threshold('THREE_COLOR', play_category, 'zhengma_multi')
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([str(num) for num in sorted(all_numbers)]),
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '正码多码')
            }
            self._add_unique_result(results, '正码多码', record)
        
        return results
    
    def _analyze_three_color_two_sides(self, account, lottery, period, group, play_category):
        """三色彩两面分析"""
        results = defaultdict(list)
        
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            bets = self.data_analyzer.extract_size_parity_from_content(content)
            
            if '大' in bets:
                has_big = True
            if '小' in bets:
                has_small = True
            if '单' in bets:
                has_single = True
            if '双' in bets:
                has_double = True
        
        conflict_types = []
        if has_big and has_small:
            conflict_types.append('大小')
        if has_single and has_double:
            conflict_types.append('单双')
        
        if conflict_types:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '矛盾类型': '、'.join(conflict_types),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面矛盾')
            }
            self._add_unique_result(results, '两面矛盾', record)
        
        return results
    
    def _analyze_three_color_wave(self, account, lottery, period, group, play_category):
        """三色彩波色分析"""
        results = defaultdict(list)
        
        # 收集该期号内所有波色投注
        all_waves = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            # 使用三色彩专用的波色提取方法
            waves = self.data_analyzer.extract_three_color_wave_from_content(content)
            all_waves.update(waves)
        
        # 检查是否在同一期号内同时投注了红波和绿波
        if '红波' in all_waves and '绿波' in all_waves:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '投注波色数': len(all_waves),
                '投注波色': sorted(list(all_waves)),
                '投注内容': f"同一期号内投注: {', '.join(sorted(all_waves))}",
                '排序权重': self._calculate_sort_weight({'投注波色数': len(all_waves)}, '色波红绿投注')
            }
            self._add_unique_result(results, '色波红绿投注', record)
        
        # 检查波色全包
        if len(all_waves) >= 3:  # 红波、绿波、紫波全包
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': play_category,
                '违规类型': '色波全包',
                '投注波色数': len(all_waves),
                '投注波色': sorted(list(all_waves)),
                '投注内容': f"色波全包: {', '.join(sorted(all_waves))}",
                '排序权重': self._calculate_sort_weight({'投注波色数': len(all_waves)}, '色波全包')
            }
            self._add_unique_result(results, '色波全包', record)
        
        return results
    
    def _analyze_three_color_tema(self, account, lottery, period, group, play_category):
        """三色彩特码分析"""
        return self._analyze_number_bets(account, lottery, period, group, play_category, 'THREE_COLOR', '特码多码')

    # ==================== 通用工具方法 ====================
    def _parse_lhc_content(self, content):
        """解析六合彩内容"""
        content_str = str(content)
        
        if '-' in content_str:
            parts = content_str.split('-', 1)
            return parts[1].strip()
        
        return content_str
    
    def _analyze_number_bets(self, account, lottery, period, group, play_category, lottery_type, result_key):
        """通用号码投注分析"""
        results = defaultdict(list)
        
        all_numbers = set()
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            
            if lottery_type == 'LHC':
                clean_content = self._parse_lhc_content(content)
            else:
                clean_content = content
                
            config = LOTTERY_CONFIGS.get(lottery_type, {})
            min_num = config.get('min_number', 0)
            max_num = config.get('max_number', 49)
            
            numbers = self.data_analyzer.extract_numbers_from_content(clean_content, min_num, max_num)
            all_numbers.update(numbers)
        
        threshold = self.threshold_manager.get_threshold(lottery_type, play_category, 'multi_number')
        if len(all_numbers) >= threshold:
            record = self._create_number_record(
                account, lottery, period, play_category, all_numbers, result_key
            )
            self._add_unique_result(results, result_key, record)
        
        return results
    
    def _create_number_record(self, account, lottery, period, play_category, numbers, result_key):
        """创建号码记录"""
        return {
            '会员账号': account,
            '彩种': lottery,
            '期号': period,
            '玩法分类': play_category,
            '号码数量': len(numbers),
            '投注内容': ', '.join([f"{num:02d}" for num in sorted(numbers)]),
            '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, result_key)
        }
    
    def _get_record_hash(self, record):
        """生成记录的唯一哈希值"""
        key_parts = [
            record['会员账号'],
            record['彩种'], 
            record['期号'],
            record.get('玩法分类', ''),
            record.get('违规类型', ''),
            record.get('位置', ''),
            str(record.get('号码数量', 0)),
            record.get('矛盾类型', '')
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _add_unique_result(self, results, result_type, record):
        """添加唯一的结果记录"""
        record_hash = self._get_record_hash(record)
        
        if record_hash not in self.seen_records:
            self.seen_records.add(record_hash)
            results[result_type].append(record)
            return True
        return False
    
    def _calculate_sort_weight(self, record, result_type):
        """计算排序权重"""
        weight = 0
        
        # 基于号码数量
        if record.get('号码数量', 0) > 0:
            weight += record['号码数量'] * 10
        
        # 基于矛盾类型复杂度
        if record.get('矛盾类型'):
            conflict_count = len(record['矛盾类型'].split('、'))
            weight += conflict_count * 15
        
        # 基于其他数量字段
        for field in ['生肖数量', '尾数数量', '投注区间数', '投注波色数', '投注五行数']:
            if record.get(field, 0) > 0:
                weight += record[field] * 8
        
        # 基于矛盾值
        if record.get('矛盾值', 0) > 0:
            weight += record['矛盾值'] * 5
        
        # 基于检测类型重要性
        if '多号码' in result_type:
            weight += 25
        elif '矛盾' in result_type:
            weight += 20
        elif '全包' in result_type:
            weight += 30
        elif '三组' in result_type:
            weight += 35
        
        return weight

# ==================== 结果处理器 ====================
class ResultProcessor:
    def __init__(self):
        self.behavior_names = {
            'PK拾赛车': {
                '超码': '超码',
                '冠亚和多码': '冠亚和多码',
                '冠亚和矛盾': '冠亚和矛盾',
                '两面矛盾': '两面矛盾',
                '独立玩法矛盾': '独立玩法矛盾',
                '前一多码': '前一多码',
                '龙虎矛盾': '龙虎矛盾',
                '十个位置全投': '十个位置全投'
            },
            '快三': {
                '和值多码': '和值多码',
                '和值矛盾': '和值矛盾',  # 大小单双同时下注
                '和值大小矛盾': '和值大小矛盾',  # 投注方向与号码分布矛盾
                '独胆多码': '独胆多码',
                '不同号全包': '不同号全包',
                '两面矛盾': '两面矛盾'
            },
            '六合彩': {
                '数字类多码': '数字类多码',
                '特码多码': '特码多码',
                '正码多码': '正码多码',
                '正码1-6多码': '正码1-6多码',
                '正特多码': '正特多码',
                '生肖类多码': '生肖类多码',
                '平特多肖': '平特多肖',
                '特肖多肖': '特肖多肖',
                '一肖多肖': '一肖多肖',
                # 尾数相关行为类型独立显示
                '尾数多码': '尾数多码',
                '尾数头尾多码': '尾数头尾多码',
                '特尾多尾': '特尾多尾',
                '全尾多尾': '全尾多尾',
                '两面玩法矛盾': '两面玩法矛盾',
                '正码1-6矛盾': '正码1-6矛盾',
                '正特矛盾': '正特矛盾',
                '区间多组': '区间多组',
                '波色三组': '波色三组',
                '色波三组': '色波三组',
                # 连肖相关 - 具体类型
                '二连肖多肖': '二连肖多肖',
                '三连肖多肖': '三连肖多肖', 
                '四连肖多肖': '四连肖多肖',
                '五连肖多肖': '五连肖多肖',
                '连肖多肖': '连肖多肖',  # 保留通用类型作为后备
                # 正码波色相关
                '正码波色全包': '正码波色全包',           
                # 连尾相关 - 具体类型
                '二连尾多尾': '二连尾多尾',
                '三连尾多尾': '三连尾多尾',
                '四连尾多尾': '四连尾多尾',
                '五连尾多尾': '五连尾多尾',
                '连尾多尾': '连尾多尾',  # 保留通用类型作为后备
                # 波色相关行为
                '色波全包': '色波全包',                   # 传统色波全包
                '七色波多色': '七色波多色',
                '色波中半波全包': '色波中半波全包',       # 色波玩法中的半波全包
                '半波大小全包': '半波大小全包',           # 半波玩法中的大小全包
                '半波单双全包': '半波单双全包',           # 半波玩法中的单双全包
                '五行多组': '五行多组',
                '连肖多肖': '连肖多肖',
                '连尾多尾': '连尾多尾'
            },
            '3D系列': {
                '两面矛盾': '两面矛盾',
                '定位胆多码': '定位胆多码'
            },
            '时时彩': {
                '两面矛盾': '两面矛盾',
                '斗牛多码': '斗牛多码',
                '定位胆多码': '定位胆多码',
                '总和矛盾': '总和矛盾'
            },
            '三色彩': {
                '正码多码': '正码多码',
                '两面矛盾': '两面矛盾',
                '色波全包': '色波全包',
                '色波红绿投注': '色波红绿投注'
            }
        }
        self.displayed_records_cache = set()  # 缓存已显示的记录
    
    def organize_results_by_account(self, all_results):
        """组织结果按账户分类本"""
        account_results = defaultdict(lambda: {
            'violations': [],
            'periods': set(),
            'violation_types': set(),
            'violation_count': 0,
            'lottery_types': set(),
            'violations_by_type': defaultdict(list),
            'violations_by_lottery': defaultdict(lambda: defaultdict(list))
        })
        
        for lottery_type, results in all_results.items():
            for result_type, records in results.items():
                for record in records:
                    account = record['会员账号']
                    period = record['期号']
                    lottery = record['彩种']
                    
                    violation_record = {
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': record['玩法分类'],
                        '违规类型': result_type,
                        '详细信息': self._get_violation_details(record, result_type),
                        '投注内容': record.get('投注内容', ''),
                        '号码数量': record.get('号码数量', 0),
                        '矛盾类型': record.get('矛盾类型', ''),
                        '位置': record.get('位置', ''),
                        '排序权重': record.get('排序权重', 0)
                    }
                    
                    account_results[account]['violations'].append(violation_record)
                    account_results[account]['violations_by_type'][result_type].append(violation_record)
                    account_results[account]['violations_by_lottery'][lottery][result_type].append(violation_record)
                    account_results[account]['periods'].add(period)
                    account_results[account]['violation_types'].add(result_type)
                    account_results[account]['violation_count'] += 1
                    account_results[account]['lottery_types'].add(lottery)
        
        return account_results
    
    def _get_violation_details(self, record, result_type):
        """获取违规详情"""
        details = []
        
        # 专门处理和值大小矛盾的显示
        if '和值大小矛盾' in result_type:
            # 和值大小矛盾显示矛盾类型和矛盾值
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            if record.get('矛盾值', 0) > 0:
                details.append(f"矛盾值: {record['矛盾值']}")
            return ' | '.join(details) if details else '无详情'
        
        # 专门处理和值矛盾的显示
        elif '和值矛盾' in result_type:
            # 和值矛盾只显示矛盾类型
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            return ' | '.join(details) if details else '无详情'
        
        # 尾数多码的特殊处理
        elif '尾数' in result_type:
            # 优先使用尾数数量，如果没有则使用号码数量
            tail_count = record.get('尾数数量', record.get('号码数量', 0))
            details.append(f"尾数数量: {tail_count}个")
        
        # 正常处理其他类型
        else:
            if '号码数量' in record and record['号码数量'] > 0:
                details.append(f"号码数量: {record['号码数量']}")
            if '矛盾类型' in record:
                details.append(f"矛盾类型: {record['矛盾类型']}")
            if '位置' in record:
                details.append(f"位置: {record['位置']}")
            if '生肖数量' in record and record['生肖数量'] > 0:
                details.append(f"生肖数量: {record['生肖数量']}")
            if '投注区间数' in record and record['投注区间数'] > 0:
                details.append(f"投注区间数: {record['投注区间数']}")
            if '投注波色数' in record and record['投注波色数'] > 0:
                details.append(f"投注波色数: {record['投注波色数']}")
            if '投注五行数' in record and record['投注五行数'] > 0:
                details.append(f"投注五行数: {record['投注五行数']}")
        
        return ' | '.join(details) if details else '无详情'
    
    def optimize_display_records(self, records, max_records=5):
        """优化显示记录"""
        if not records:
            return []
        
        # 重置缓存（每次调用时重新计算）
        self.displayed_records_cache = set()
        
        def get_record_key(record):
            """生成记录的唯一键"""
            return (
                record.get('会员账号', ''),
                record.get('期号', ''),
                record.get('玩法分类', ''),
                record.get('违规类型', ''),
                record.get('位置', ''),
                record.get('矛盾类型', '')
            )
        
        # 去重并排序
        unique_records = []
        seen_keys = set()
        
        for record in records:
            record_key = get_record_key(record)
            if record_key not in seen_keys:
                seen_keys.add(record_key)
                unique_records.append(record)
        
        # 按排序权重排序
        unique_records.sort(key=lambda x: x.get('排序权重', 0), reverse=True)
        
        # 对于和值矛盾，确保展示多样性
        if unique_records and '和值矛盾' in unique_records[0].get('违规类型', ''):
            return self._ensure_variety_in_display(unique_records, max_records)
        else:
            return unique_records[:max_records]
    
    def _ensure_variety_in_display(self, records, max_records=5):
        """确保展示的记录包含不同类型的矛盾"""
        if len(records) <= max_records:
            return records
        
        # 按矛盾类型分组
        conflict_groups = {
            '大小': [],
            '单双': [], 
            '大小单双': [],
            '其他': []
        }
        
        for record in records:
            conflict_type = record.get('矛盾类型', '')
            if '大小' in conflict_type and '单双' in conflict_type:
                conflict_groups['大小单双'].append(record)
            elif '大小' in conflict_type:
                conflict_groups['大小'].append(record)
            elif '单双' in conflict_type:
                conflict_groups['单双'].append(record)
            else:
                conflict_groups['其他'].append(record)
        
        # 优先从每个类型中选取代表性记录
        selected_records = []
        
        # 第一轮：从每个非空类型中各取1条
        for group_name in ['大小单双', '大小', '单双', '其他']:
            if conflict_groups[group_name] and len(selected_records) < max_records:
                selected_records.append(conflict_groups[group_name][0])
        
        # 如果还没取满，继续按原有顺序补充
        if len(selected_records) < max_records:
            # 获取已选记录的索引，避免重复
            selected_indices = set(records.index(r) for r in selected_records)
            
            for record in records:
                if records.index(record) not in selected_indices and len(selected_records) < max_records:
                    selected_records.append(record)
        
        return selected_records
    
    def create_summary_stats(self, account_results, df_clean):
        """创建汇总统计"""
        total_violations = sum(data['violation_count'] for data in account_results.values())
        
        summary = {
            '总记录数': len(df_clean),
            '总会员数': df_clean['会员账号'].nunique(),
            '违规账户数': len(account_results),
            '总违规记录数': total_violations,
            '总违规期数': sum(len(data['periods']) for data in account_results.values()),
            '彩种分布': df_clean['彩种'].value_counts().to_dict(),
            '违规类型统计': defaultdict(int),
            '账户违规统计': []
        }
        
        for account, data in account_results.items():
            for violation_type in data['violation_types']:
                summary['违规类型统计'][violation_type] += len(data['violations_by_type'][violation_type])
            
            summary['账户违规统计'].append({
                '账户': account,
                '违规期数': len(data['periods']),
                '违规次数': data['violation_count'],
                '违规类型数': len(data['violation_types']),
                '彩种数': len(data['lottery_types'])
            })
        
        summary['账户违规统计'] = sorted(summary['账户违规统计'], key=lambda x: x['违规次数'], reverse=True)
        
        return summary
    
    def display_summary(self, summary):
        """显示汇总统计"""
        st.subheader("📊 汇总统计")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总记录数", summary['总记录数'])
        with col2:
            st.metric("总会员数", summary['总会员数'])
        with col3:
            st.metric("违规账户数", summary['违规账户数'])
        with col4:
            st.metric("总违规记录数", summary['总违规记录数'])
        
        if summary['违规类型统计']:
            with st.expander("📈 违规类型分布", expanded=False):
                violation_df = pd.DataFrame({
                    '违规类型': list(summary['违规类型统计'].keys()),
                    '数量': list(summary['违规类型统计'].values())
                }).sort_values('数量', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.bar_chart(violation_df.set_index('违规类型'))
                with col2:
                    st.dataframe(violation_df, hide_index=True)
        
        if summary['账户违规统计']:
            with st.expander("🏆 账户违规排名", expanded=False):
                top_accounts = summary['账户违规统计'][:10]
                account_df = pd.DataFrame(top_accounts)
                st.dataframe(account_df, hide_index=True)
    
    def display_account_results(self, account_results):
        """显示账户结果本"""
        if not account_results:
            st.info("🎉 未发现可疑投注行为")
            return
        
        st.subheader("🔍 违规账户详情")
        
        sorted_accounts = sorted(account_results.items(), 
                               key=lambda x: x[1]['violation_count'], 
                               reverse=True)
        
        for account_index, (account, data) in enumerate(sorted_accounts, 1):
            # 转义账号中的下划线
            account_display = account.replace('_', '\\_')
            
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.subheader(f"{account_index}. {account_display}")  # 使用转义后的账号
                    # 使用 data 中的 lottery_types
                    lottery_types_list = list(data['lottery_types'])
                    st.write(f"**涉及彩种:** {', '.join(lottery_types_list[:5])}{'...' if len(lottery_types_list) > 5 else ''}")

                with col2:
                    # 使用 data 中的 violation_types
                    violation_types_list = list(data['violation_types'])
                    violation_text = "、".join(violation_types_list[:5])
                    if len(violation_types_list) > 5:
                        violation_text += f" 等{len(violation_types_list)}种"
                    st.write(f"**违规内容:** {violation_text}")

                with col3:
                    # 使用 data 中的 periods 和 violation_count
                    st.write(f"**违规期数:** {len(data['periods'])}")
                    st.write(f"**违规次数:** {data['violation_count']}")
                
                # 按彩种和违规类型分组显示，避免重复
                displayed_violations = set()
                
                for lottery in sorted(data['violations_by_lottery'].keys()):
                    lottery_violations = data['violations_by_lottery'][lottery]
                    
                    with st.expander(f"🎯 {lottery} (共{sum(len(v) for v in lottery_violations.values())}次违规)", expanded=True):
                        
                        for violation_type in sorted(lottery_violations.keys()):
                            type_violations = lottery_violations[violation_type]
                            
                            # 使用优化显示方法
                            representative_records = self.optimize_display_records(type_violations, max_records=5)
                            other_records_count = len(type_violations) - len(representative_records)
                            
                            if representative_records:
                                st.write(f"**{violation_type}** ({len(type_violations)}次)")
                                
                                # 准备显示数据
                                display_data = []
                                for record in representative_records:
                                    display_record = {
                                        '期号': record['期号'],
                                        '玩法分类': record['玩法分类'],
                                        '违规类型': violation_type,
                                        '详细信息': record.get('详细信息', ''),
                                        '投注内容': record.get('投注内容', '')
                                    }
                                    # 添加位置信息（如果有）
                                    if record.get('位置'):
                                        display_record['位置'] = record['位置']
                                    display_data.append(display_record)
                                
                                df_display = pd.DataFrame(display_data)
                                container = st.container()
                                with container:
                                    st.dataframe(
                                        df_display,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=min(300, len(representative_records) * 35 + 38)
                                    )
                                
                                if other_records_count > 0:
                                    st.info(f"还有 {other_records_count} 条相关记录...")
                
                st.markdown("---")

# ==================== 导出功能 ====================
class Exporter:
    """结果导出器"""
    
    def prepare_export_data(self, account_summary):
        """准备导出数据"""
        export_data = []
        
        for account, summary in account_summary.items():
            for lottery, lottery_data in summary['violations_by_lottery'].items():
                for behavior_type, records in lottery_data.items():
                    for record in records:
                        export_record = {
                            '会员账号': account,
                            '彩种': lottery,
                            '期号': record['期号'],
                            '玩法分类': record['玩法分类'],
                            '行为类型': behavior_type
                        }
                        
                        # 添加矛盾类型
                        if '矛盾类型' in record:
                            export_record['矛盾类型'] = record['矛盾类型']
                        
                        # 添加数量信息
                        self._add_quantity_info(export_record, record, behavior_type)
                        
                        export_data.append(export_record)
        
        return export_data
    
    def _add_quantity_info(self, export_record, record, behavior_type):
        """添加数量信息到导出记录"""
        quantity_fields = {
            # 快三相关
            '和值多码': ('号码数量', '投注内容'),
            '和值矛盾': (None, '投注内容'),  # 和值矛盾只有投注内容
            '和值大小矛盾': ('矛盾值', '投注内容'),  # 和值大小矛盾有矛盾值
            '独胆多码': ('号码数量', '投注内容'),
            '不同号全包': ('号码数量', '投注内容'),
            '两面矛盾': (None, '投注内容'),

            # PK10新增
            '十个位置全投': ('投注位置数', '投注内容'),
            
            # 六合彩相关
            '数字类多码': ('号码数量', '投注内容'),
            '特码多码': ('号码数量', '投注内容'),
            '正码多码': ('号码数量', '投注内容'),
            '正码1-6多码': ('号码数量', '投注内容'),
            '正特多码': ('号码数量', '投注内容'),
            '生肖类多码': ('生肖数量', '投注内容'),
            '平特多肖': ('生肖数量', '投注内容'),
            '特肖多肖': ('生肖数量', '投注内容'),
            '一肖多肖': ('生肖数量', '投注内容'),
            '尾数多码': ('尾数数量', '投注内容'),
            '尾数头尾多码': ('尾数数量', '投注内容'),
            '特尾多尾': ('尾数数量', '投注内容'),
            '全尾多尾': ('尾数数量', '投注内容'),
            '连肖多肖': ('生肖数量', '投注内容'),
            '连尾多尾': ('尾数数量', '投注内容'),
            '区间多组': ('投注区间数', '投注内容'),
            '波色三组': ('投注波色数', '投注内容'),
            '色波三组': ('投注波色数', '投注内容'),
            # 修改为只记录全包情况
            '色波全包': ('投注波色数', '投注内容'),
            '半波单双全包': ('投注半波数', '投注内容'),
            '半波大小全包': ('投注半波数', '投注内容'),
            '五行多组': ('投注五行数', '投注内容'),
            '两面玩法矛盾': (None, '投注内容'),
            '正码1-6矛盾': (None, '投注内容'),
            '正特矛盾': (None, '投注内容'),

            # 半波相关
            '半波全包': (None, '投注内容'),
            '半波多组投注': ('投注波色数', '投注内容'),       
            
             # 三色彩相关
            '色波全包': ('投注波色数', '投注内容'),
            '色波红绿投注': ('投注波色数', '投注内容'),

             # 3D系列相关
            '两面矛盾': (None, '投注内容'),
            '定位胆多码': ('号码数量', '投注内容'),

             # 时时彩相关
            '斗牛多码': ('号码数量', '投注内容'),
            '定位胆多码': ('号码数量', '投注内容'),
            
            # PK10相关
            '超码': ('号码数量', '投注内容'),
            '冠亚和多码': ('号码数量', '投注内容'),
            '前一多码': ('号码数量', '投注内容'),
            '冠亚和矛盾': (None, '投注内容'),
            '两面矛盾': (None, '投注内容'),
            '独立玩法矛盾': (None, '投注内容'),
            '龙虎矛盾': (None, '投注内容'),
            '总和矛盾': (None, '投注内容'),
            '色波矛盾投注': (None, '投注内容'),
            '两面玩法矛盾': (None, '投注内容'),
            '正码1-6矛盾': (None, '投注内容'),
            '正特矛盾': (None, '投注内容'),
        }
        
        if behavior_type in quantity_fields:
            count_field, content_field = quantity_fields[behavior_type]
            if count_field and count_field in record:
                export_record[count_field] = record[count_field]
            
            if content_field and record.get(content_field):
                export_record[content_field] = str(record[content_field])
            
            # 添加位置信息（3D系列专用）
            if record.get('位置'):
                export_record['位置'] = record['位置']
    
    def export_to_excel(self, account_summary, filename_prefix="彩票分析结果"):
        """导出分析结果到Excel文件"""
        try:
            export_data = self.prepare_export_data(account_summary)
            
            if not export_data:
                st.warning("没有可导出的数据")
                return
            
            # 创建DataFrame
            df_export = pd.DataFrame(export_data)
            
            # 生成文件名
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{filename_prefix}_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                # 写入详细数据
                df_export.to_excel(writer, sheet_name='详细分析结果', index=False)
                
                # 创建统计工作表
                self._create_summary_sheets(writer, account_summary, export_data)
            
            st.success(f"✅ 分析结果已成功导出到: {output_filename}")
            st.info(f"📊 导出内容包含 {len(export_data)} 条记录")
            
            # 提供下载
            with open(output_filename, "rb") as file:
                btn = st.download_button(
                    label="📥 下载分析结果",
                    data=file,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        except Exception as e:
            st.error(f"❌ 导出过程中出现错误: {str(e)}")
    
    def _create_summary_sheets(self, writer, account_summary, export_data):
        """创建统计工作表"""
        # 账户统计
        account_stats = []
        for account, summary in account_summary.items():
            account_stats.append({
                '会员账号': account,
                '总可疑期号数': len(summary['periods']),
                '涉及彩种数': len(summary['lottery_types']),
                '行为类型数': len(summary['violation_types'])
            })
        
        if account_stats:
            df_account_stats = pd.DataFrame(account_stats)
            df_account_stats.to_excel(writer, sheet_name='账户统计', index=False)
        
        # 行为类型统计
        if export_data:
            behavior_stats = pd.DataFrame(export_data)['行为类型'].value_counts().reset_index()
            behavior_stats.columns = ['行为类型', '记录数']
            behavior_stats.to_excel(writer, sheet_name='行为类型统计', index=False)
        
        # 彩种统计
        if export_data:
            lottery_stats = pd.DataFrame(export_data)['彩种'].value_counts().reset_index()
            lottery_stats.columns = ['彩种', '记录数']
            lottery_stats.to_excel(writer, sheet_name='彩种统计', index=False)

# ==================== Streamlit界面 ====================
def main():
    st.title("🎯 智能彩票分析检测系统")
    st.markdown("---")
    
    st.sidebar.title("系统配置")
    
    uploaded_file = st.sidebar.file_uploader(
        "上传Excel文件", 
        type=['xlsx', 'xls'],
        help="请上传包含彩票投注数据的Excel文件"
    )
    
    st.sidebar.subheader("检测阈值配置")
    
    with st.sidebar.expander("PK拾系列阈值"):
        pk10_multi = st.slider("超码阈值", 5, 15, THRESHOLD_CONFIG['PK10']['multi_number'])
        pk10_gyh = st.slider("冠亚和多码阈值", 8, 20, THRESHOLD_CONFIG['PK10']['gyh_multi_number'])
        THRESHOLD_CONFIG['PK10']['multi_number'] = pk10_multi
        THRESHOLD_CONFIG['PK10']['gyh_multi_number'] = pk10_gyh
    
    with st.sidebar.expander("时时彩系列阈值"):
        ssc_dingwei = st.slider("定位胆多码阈值", 5, 15, THRESHOLD_CONFIG['SSC']['dingwei_multi'])
        ssc_douniu = st.slider("斗牛多码阈值", 5, 15, THRESHOLD_CONFIG['SSC']['douniu_multi'])
        THRESHOLD_CONFIG['SSC']['dingwei_multi'] = ssc_dingwei
        THRESHOLD_CONFIG['SSC']['douniu_multi'] = ssc_douniu
    
    with st.sidebar.expander("六合彩系列阈值"):
        lhc_numbers = st.slider("数字类多码阈值", 20, 50, THRESHOLD_CONFIG['LHC']['number_play'])
        lhc_zodiacs = st.slider("生肖类多码阈值", 5, 15, THRESHOLD_CONFIG['LHC']['zodiac_play'])
        lhc_tails = st.slider("尾数多码阈值", 5, 15, THRESHOLD_CONFIG['LHC']['tail_play'])
        THRESHOLD_CONFIG['LHC']['number_play'] = lhc_numbers
        THRESHOLD_CONFIG['LHC']['zodiac_play'] = lhc_zodiacs
        THRESHOLD_CONFIG['LHC']['tail_play'] = lhc_tails
    
    with st.sidebar.expander("快三系列阈值"):
        k3_hezhi = st.slider("和值多码阈值", 5, 20, THRESHOLD_CONFIG['K3']['hezhi_multi_number'])
        k3_dudan_threshold = st.slider("独胆多码阈值", 2, 6, 5)
        THRESHOLD_CONFIG['K3']['hezhi_multi_number'] = k3_hezhi
        THRESHOLD_CONFIG['K3']['dudan_multi_number'] = k3_dudan_threshold
    
    with st.sidebar.expander("三色彩系列阈值"):
        three_color_zhengma = st.slider("正码多码阈值", 5, 15, THRESHOLD_CONFIG['THREE_COLOR']['zhengma_multi'])
        THRESHOLD_CONFIG['THREE_COLOR']['zhengma_multi'] = three_color_zhengma

    with st.sidebar.expander("3D系列阈值"):
        three_d_dingwei = st.slider("3D定位胆多码阈值", 5, 10, THRESHOLD_CONFIG['3D']['dingwei_multi'])
        THRESHOLD_CONFIG['3D']['dingwei_multi'] = three_d_dingwei
    
    if uploaded_file is not None:
        try:
            with st.spinner('正在处理数据...'):
                # 初始化组件
                processor = DataProcessor()
                analyzer = UnifiedAnalyzer()
                result_processor = ResultProcessor()
                exporter = Exporter()
                
                # 数据清洗
                df_clean = processor.clean_data(uploaded_file)
                
                if df_clean is not None and len(df_clean) > 0:
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("总记录数", len(df_clean))
                    with col2:
                        st.metric("唯一会员数", df_clean['会员账号'].nunique())
                    with col3:
                        st.metric("彩种数量", df_clean['彩种'].nunique())
                    
                    # 统一玩法分类
                    play_normalizer = PlayCategoryNormalizer()
                    if '玩法' in df_clean.columns:
                        df_clean['玩法分类'] = df_clean['玩法'].apply(play_normalizer.normalize_categ
                    
                    # 分析投注模式 - 使用新的统一分析器
                    all_results = analyzer.analyze_all_patterns(df_clean)
                    # 使用进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = {}
                    # 明确定义 lottery_types 变量 - 添加3D系列
                    lottery_types = ['PK拾赛车', '时时彩', '六合彩', '快三', '三色彩', '3D系列']
                    
                    for i, lottery_type in enumerate(lottery_types):
                        status_text.text(f"正在分析 {lottery_type}...")
                        
                        if lottery_type == 'PK拾赛车':
                            all_results[lottery_type] = analyzer.analyze_pk10_patterns(df_normalized)
                        elif lottery_type == '时时彩':
                            all_results[lottery_type] = analyzer.analyze_ssc_patterns(df_normalized)
                        elif lottery_type == '六合彩':
                            all_results[lottery_type] = analyzer.analyze_lhc_patterns(df_normalized)
                        elif lottery_type == '快三':
                            all_results[lottery_type] = analyzer.analyze_k3_patterns(df_normalized)
                        elif lottery_type == '三色彩':
                            all_results[lottery_type] = analyzer.analyze_three_color_patterns(df_normalized)
                        # 添加3D系列分析调用
                        elif lottery_type == '3D系列':
                            all_results[lottery_type] = analyzer.analyze_3d_patterns(df_normalized)
                        
                        progress_bar.progress((i + 1) / len(lottery_types))
                    
                    status_text.text("分析完成！")
                    
                    # 统计结果
                    total_findings = 0
                    for lottery_type, results in all_results.items():
                        type_count = sum(len(records) for records in results.values())
                        total_findings += type_count
                    
                    with col4:
                        st.metric("可疑记录数", total_findings)
                    
                    with st.expander("📊 数据预览", expanded=False):
                        st.dataframe(df_clean.head(10))
                    
                    if total_findings == 0:
                        st.success("🎉 未发现可疑投注行为")
                    else:
                        # 处理并显示结果
                        account_results = result_processor.organize_results_by_account(all_results)
                        
                        summary_stats = result_processor.create_summary_stats(account_results, df_clean)
                        result_processor.display_summary(summary_stats)
                        
                        result_processor.display_account_results(account_results)
                        
                        # 导出结果
                        st.subheader("📥 结果导出")
                        exporter.export_to_excel(account_results, "智能彩票分析")
                
                else:
                    st.error("❌ 数据清洗后无有效数据，请检查文件格式")
        
        except Exception as e:
            st.error(f"❌ 处理过程中出现错误: {str(e)}")
            logger.error(f"处理过程中出现错误: {str(e)}")
    
    else:
        st.markdown("""
        ## 📋 使用说明
        
        1. **上传文件**: 在左侧边栏上传Excel格式的彩票投注数据文件
        2. **配置阈值**: 根据需要调整各类彩票的检测阈值
        3. **查看结果**: 系统将自动分析并显示可疑投注行为
        4. **导出结果**: 下载详细的检测报告
        
        ### 🎯 系统特色
        
        **🔍 全面检测能力**
        - ✅ PK拾/赛车系列：超码、冠亚和矛盾、两面矛盾、龙虎矛盾
        - ✅ 时时彩系列：定位胆多码、斗牛多码、两面矛盾、总和矛盾  
        - ✅ 六合彩系列：特码/正码多码、生肖多号码、尾数多码、波色五行矛盾
        - ✅ 快三系列：和值多码、和值矛盾、和值大小矛盾、独胆多码、不同号全包、两面矛盾
        - ✅ 三色彩系列：正码多码、两面矛盾、色波矛盾
        
        **🚀 技术优势**
        - 📊 完整的尾数检测
        - ⚡ 缓存优化的号码提取算法
        - 🎯 智能的玩法分类映射
        - 📈 详细的数据质量验证
        - 🔄 实时进度显示和性能监控
        
        **💡 用户体验**
        - 🎨 现代化的Streamlit界面
        - ⚙️ 实时可调的检测阈值
        - 📱 响应式布局设计
        - 📥 一键导出完整报告
        
        ### 📝 支持的数据格式
        
        系统会自动识别以下列名变体：
        
        - **会员账号**: 会员账号、会员账户、账号、账户、用户账号、玩家账号、用户ID、玩家ID
        - **彩种**: 彩种、彩神、彩票种类、游戏类型、彩票类型、游戏彩种、彩票名称
        - **期号**: 期号、期数、期次、期、奖期、期号信息、期号编号
        - **玩法**: 玩法、玩法分类、投注类型、类型、投注玩法、玩法类型、分类
        - **内容**: 内容、投注内容、下注内容、注单内容、投注号码、号码内容、投注信息
        - **金额**: 金额、下注总额、投注金额、总额、下注金额、投注额、金额数值
        
        ### 🎲 支持的彩种
        
        **PK拾/赛车系列**
        - 分分PK拾、三分PK拾、五分PK拾、新幸运飞艇、澳洲幸运10
        - 一分PK10、宾果PK10、极速飞艇、澳洲飞艇、幸运赛车
        - 分分赛车、北京PK10、旧北京PK10、极速赛车、幸运赛車、北京赛车、极速PK10、幸运PK10、赛车、赛車
        
        **时时彩系列**
        - 分分时时彩、三分时时彩、五分时时彩、宾果时时彩
        - 1分时时彩、3分时时彩、5分时时彩、旧重庆时时彩
        - 幸运时时彩、腾讯分分彩、新疆时时彩、天津时时彩、重庆时时彩、上海时时彩、广东时时彩、分分彩、时时彩、時時彩
        
        **六合彩系列**
        - 新澳门六合彩、澳门六合彩、香港六合彩、一分六合彩
        - 五分六合彩、三分六合彩、香港⑥合彩、分分六合彩
        - 快乐6合彩、港⑥合彩、台湾大乐透、六合、lhc、六合彩、⑥合、6合
        
        **快三系列**
        - 分分快三、三分快3、五分快3、澳洲快三、宾果快三
        - 1分快三、3分快三、5分快三、10分快三、加州快三
        - 幸运快三、大发快三、快三、快3、k3、k三

        **3D系列**
        - 排列三、排列3、幸运排列3、一分排列3、二分排列3、三分排列3
        - 五分排列3、十分排列3、大发排列3、好运排列3、福彩3D、极速3D
        - 极速排列3、幸运3D、一分3D、二分3D、三分3D、五分3D、十分3D、大发3D、好运3D
        
        **三色彩系列**
        - 一分三色彩、30秒三色彩、五分三色彩、三分三色彩、三色、三色彩、三色球
        
        ---
        
        **注意**: 请确保上传的Excel文件包含必要的列信息，系统会自动识别常见的列名变体。
        
        """)

# 确保主函数被调用
if __name__ == "__main__":
    main()
