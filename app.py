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
        'all_positions_bet': 10,
        'same_number_multi_position': 7
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
        'five_elements': 4,
        'value_size_contradiction': 7,  # 新增：六合彩特码变相超码阈值
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
        """智能列识别 - 隐藏详情版本"""
        identified_columns = {}
        actual_columns = [str(col).strip() for col in df_columns]

        for standard_col, possible_names in self.column_mapping.items():
            found = False
            for actual_col in actual_columns:
                actual_col_lower = actual_col.lower().replace(' ', '').replace('_', '').replace('-', '')
                
                for possible_name in possible_names:
                    possible_name_lower = possible_name.lower().replace(' ', '').replace('_', '').replace('-', '')
                    
                    # 使用相似度算法
                    similarity = len(set(possible_name_lower) & set(actual_col_lower)) / len(possible_name_lower)
                    if similarity > 0.7:  # 70%相似度阈值
                        identified_columns[actual_col] = standard_col
                        found = True
                        break
                
                if found:
                    break

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
        """数据质量验证 - 隐藏所有输出版本"""
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
            # 截断账号检测
            truncated_accounts = df[df['会员账号'].str.contains(r'\.\.\.|…', na=False)]
            if len(truncated_accounts) > 0:
                issues.append(f"发现 {len(truncated_accounts)} 个可能被截断的会员账号")
            
            # 账号长度异常检测
            account_lengths = df['会员账号'].str.len()
            if account_lengths.max() > 50:  # 假设正常账号长度不超过50个字符
                issues.append("发现异常长度的会员账号")

        # 检查数据类型
        if '期号' in df.columns:
            # 确保期号为字符串类型
            df['期号'] = df['期号'].astype(str)
            # 修复期号格式问题：去掉.0
            df['期号'] = df['期号'].str.replace(r'\.0$', '', regex=True)
            # 允许期号包含字母和数字
            invalid_periods = df[~df['期号'].str.match(r'^[\dA-Za-z]+$', na=True)]
            if len(invalid_periods) > 0:
                issues.append(f"发现 {len(invalid_periods)} 条无效期号记录")
        
        # 检查金额列的有效性
        if '金额' in df.columns:
            try:
                # 尝试转换为数值类型
                df['金额'] = pd.to_numeric(df['金额'], errors='coerce')
                invalid_amounts = df['金额'].isnull().sum()
                if invalid_amounts > 0:
                    issues.append(f"发现 {invalid_amounts} 条无效金额记录")
            except Exception as e:
                issues.append(f"金额列转换失败: {str(e)}")
        
        # 检查重复数据
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"发现 {duplicate_count} 条重复记录")

        return issues

    @staticmethod
    def enhanced_extract_amount(amount_text):
        """增强金额提取 - 支持多种格式"""
        try:
            if pd.isna(amount_text) or amount_text is None:
                return 0.0
            
            text = str(amount_text).strip()
            if text == '':
                return 0.0
            
            # 方法1: 直接转换（处理纯数字）
            try:
                # 移除所有非数字字符（除了点和负号）
                clean_text = re.sub(r'[^\d.-]', '', text)
                if clean_text and clean_text != '-' and clean_text != '.':
                    amount = float(clean_text)
                    if amount >= 0:
                        return amount
            except:
                pass
            
            # 方法2: 处理千位分隔符格式
            try:
                # 移除逗号和全角逗号，然后转换
                clean_text = text.replace(',', '').replace('，', '')
                amount = float(clean_text)
                if amount >= 0:
                    return amount
            except:
                pass
            
            # 方法3: 处理"5.000"这种格式
            if re.match(r'^\d+\.\d{3}$', text):
                try:
                    amount = float(text)
                    return amount
                except:
                    pass
            
            # 方法4: 使用正则表达式提取各种格式
            patterns = [
                r'投注\s*[:：]?\s*([\d,.]+)',
                r'金额\s*[:：]?\s*([\d,.]+)',
                r'下注金额\s*([\d,.]+)',
                r'([\d,.]+)\s*元',
                r'￥\s*([\d,.]+)',
                r'¥\s*([\d,.]+)',
                r'([\d,.]+)\s*RMB',
                r'([\d,.]+)$'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '').replace('，', '')
                    try:
                        amount = float(amount_str)
                        if amount >= 0:
                            return amount
                    except:
                        continue
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"金额提取失败: {amount_text}, 错误: {str(e)}")
            return 0.0
    
    def clean_data(self, uploaded_file):
        """数据清洗主函数 - 修改版本：隐藏所有中间过程信息"""
        try:
            # 第一次读取用于定位
            df_temp = pd.read_excel(uploaded_file, header=None, nrows=50)
            
            # 找到数据起始位置
            start_row, start_col = self.find_data_start(df_temp)
            
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
            
            # 智能列识别 - 隐藏详情
            column_mapping = self.smart_column_identification(df_clean.columns)
            if column_mapping:
                df_clean = df_clean.rename(columns=column_mapping)
                for old_col, new_col in column_mapping.items():
                    logger.info(f"  {old_col} -> {new_col}")
            
            # 确保必要列存在
            missing_columns = [col for col in self.required_columns if col not in df_clean.columns]
            if missing_columns and len(df_clean.columns) >= 4:
                # 隐藏自动映射消息
                manual_mapping = {}
                col_names = ['会员账号', '彩种', '期号', '内容', '玩法', '金额']
                for i, col_name in enumerate(col_names):
                    if i < len(df_clean.columns):
                        manual_mapping[df_clean.columns[i]] = col_name
                
                df_clean = df_clean.rename(columns=manual_mapping)
            
            # 数据清理
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[col for col in self.required_columns if col in df_clean.columns])
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # 添加特殊字符处理
            for col in df_clean.columns:
                if col in ['玩法', '内容']:  # 特别处理这些列
                    df_clean[col] = df_clean[col].apply(
                        lambda x: self.normalize_special_characters(str(x)) if pd.notna(x) else x
                    )
            
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
            
            # 修复期号格式：去掉.0 - 改进：确保转换为字符串
            if '期号' in df_clean.columns:
                df_clean['期号'] = df_clean['期号'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            # 验证金额列的有效性
            if '金额' in df_clean.columns:
                try:
                    # 使用增强金额提取
                    df_clean['金额'] = df_clean['金额'].apply(self.enhanced_extract_amount)
                    invalid_amounts = df_clean['金额'].isnull().sum()
                    if invalid_amounts > 0:
                        pass
                except Exception as e:
                    logger.debug(f"金额转换失败: {str(e)}")
         
            # 数据质量验证 - 隐藏所有输出
            issues = self.validate_data_quality(df_clean)
        
            return df_clean
            
        except Exception as e:
            st.error(f"❌ 数据清洗失败: {str(e)}")
            logger.error(f"数据清洗失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None

    def normalize_special_characters(self, text):
        """规范化特殊字符，特别是空格字符"""
        import re
        if not text:
            return text
        
        # 将所有类型的空白字符（包括不间断空格）替换为普通空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text

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
        try:
            content_str = str(content).strip()
            bets_by_position = defaultdict(list)
            
            if not content_str:
                return bets_by_position
            
            # 定义位置映射 - 修正重复的位置
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
                        # 单个数字 - 修复：使用part_clean
                        if part_clean.isdigit():
                            numbers.append(int(part_clean))
                    
                    # 添加到对应位置
                    bets_by_position[position].extend(numbers)
            
            return bets_by_position
        except Exception as e:
            logger.warning(f"解析PK10竖线格式失败: {content}, 错误: {str(e)}")
            return defaultdict(list)
    
    @staticmethod
    def parse_ssc_vertical_format(content):
        """
        解析时时彩竖线分隔的定位胆格式
        格式：号码1,号码2|号码3|号码4,号码5|号码6|号码7,号码8,号码9|号码10
        或者：_|05|_|_|_ 表示只有第二个位置有投注
        """
        try:
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
                        # 单个数字 - 修复：使用part_clean
                        if part_clean.isdigit():
                            numbers.append(int(part_clean))
                    
                    # 添加到对应位置
                    bets_by_position[position].extend(numbers)
            
            return bets_by_position
        except Exception as e:
            logger.warning(f"解析时时彩竖线格式失败: {content}, 错误: {str(e)}")
            return defaultdict(list)

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
        try:
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
                        # 单个数字 - 修复：使用part_clean
                        if part_clean.isdigit():
                            numbers.append(int(part_clean))
                    
                    # 添加到对应位置
                    bets_by_position[position].extend(numbers)
            
            return bets_by_position
        except Exception as e:
            logger.warning(f"解析3D竖线格式失败: {content}, 错误: {str(e)}")
            return defaultdict(list)

# ==================== 增强内容解析器 ====================
class EnhancedContentParser:
    """增强版内容解析器，支持从内容中识别玩法和位置"""
    
    @staticmethod
    def extract_play_method_and_position(content, lottery_type):
        """
        从投注内容中提取玩法和位置
        返回: (玩法类型, 位置, 纯投注内容)
        """
        content_str = str(content).strip()
        
        if lottery_type == 'LHC':
            return EnhancedContentParser._parse_lhc_content(content_str)
        elif lottery_type == 'PK10':
            return EnhancedContentParser._parse_pk10_content(content_str)
        elif lottery_type == 'SSC':
            return EnhancedContentParser._parse_ssc_content(content_str)
        elif lottery_type == '3D':
            return EnhancedContentParser._parse_3d_content(content_str)
        else:
            return '未知玩法', '未知位置', content_str
    
    @staticmethod
    def _parse_lhc_content(content):
        """解析六合彩内容"""
        # 正码位置识别
        zhengma_positions = {
            '正码一': ['正码一', '正码1', '正1', '正一'],
            '正码二': ['正码二', '正码2', '正2', '正二'],
            '正码三': ['正码三', '正码3', '正3', '正三'],
            '正码四': ['正码四', '正码4', '正4', '正四'],
            '正码五': ['正码五', '正码5', '正5', '正五'],
            '正码六': ['正码六', '正码6', '正6', '正六']
        }
        
        # 玩法类型识别
        play_methods = {
            '波色': ['红波', '蓝波', '绿波', '波色'],
            '大小': ['大', '小'],
            '单双': ['单', '双'],
            '尾数': ['尾'],
            '生肖': ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
        }
        
        position = '未知位置'
        play_method = '未知玩法'
        clean_content = content
        
        # 识别位置
        for pos_name, keywords in zhengma_positions.items():
            for keyword in keywords:
                if keyword in content:
                    position = pos_name
                    # 从内容中移除位置信息，得到纯投注内容
                    clean_content = clean_content.replace(keyword, '').strip()
                    break
            if position != '未知位置':
                break
        
        # 识别玩法
        for method_name, keywords in play_methods.items():
            for keyword in keywords:
                if keyword in clean_content:
                    play_method = method_name
                    break
            if play_method != '未知玩法':
                break
        
        return play_method, position, clean_content

    @staticmethod
    def _parse_pk10_content(content):
        """解析PK10内容 - 专门修复冠军亚军季军"""
        # 处理特殊空格字符
        import re
        content_processed = re.sub(r'[ \t\u00A0\u3000\u2000-\u200B]+', ' ', str(content).strip())
        
        # PK10位置识别 - 专门处理冠军亚军季军
        pk10_positions = {
            '冠军': ['冠军', '第1名', '第一名', '前一', '冠 军', '冠　军'],
            '亚军': ['亚军', '第2名', '第二名', '亚 军', '亚　军'],
            '季军': ['季军', '第3名', '第三名', '季 军', '季　军'],
            '第四名': ['第四名', '第4名'],
            '第五名': ['第五名', '第5名'],
            '第六名': ['第六名', '第6名'],
            '第七名': ['第七名', '第7名'],
            '第八名': ['第八名', '第8名'],
            '第九名': ['第九名', '第9名'],
            '第十名': ['第十名', '第10名']
        }
        
        position = '未知位置'
        play_method = '未知玩法'
        clean_content = content_processed
        
        # 专门处理冠军、亚军、季军的识别
        # 首先检查是否包含位置关键词
        for pos_name, keywords in pk10_positions.items():
            for keyword in keywords:
                # 使用处理后的内容进行匹配
                if keyword in content_processed:
                    position = pos_name
                    # 从内容中移除位置信息，得到纯投注内容
                    clean_content = clean_content.replace(keyword, '').strip()
                    break
            if position != '未知位置':
                break
        
        # 如果仍然没有识别到位置，尝试更宽松的匹配
        if position == '未知位置':
            if any(word in content_processed for word in ['冠军', '冠 军', '冠　军', '冠  军']):
                position = '冠军'
            elif any(word in content_processed for word in ['亚军', '亚 军', '亚　军', '亚  军']):
                position = '亚军' 
            elif any(word in content_processed for word in ['季军', '季 军', '季　军', '季  军']):
                position = '季军'
        
        # 识别玩法类型
        play_methods = {
            '龙虎': ['龙', '虎'],
            '大小': ['大', '小'],
            '单双': ['单', '双']
        }
        
        for method_name, keywords in play_methods.items():
            for keyword in keywords:
                if keyword in clean_content:
                    play_method = method_name
                    break
            if play_method != '未知玩法':
                break
        
        return play_method, position, clean_content

    @staticmethod
    def _parse_ssc_content(content):
        """解析时时彩内容"""
        # 时时彩位置识别
        ssc_positions = {
            '第1球': ['第1球', '万位', '第一位'],
            '第2球': ['第2球', '千位', '第二位'],
            '第3球': ['第3球', '百位', '第三位'],
            '第4球': ['第4球', '十位', '第四位'],
            '第5球': ['第5球', '个位', '第五位']
        }
        
        # 玩法类型识别
        play_methods = {
            '大小': ['大', '小'],
            '单双': ['单', '双'],
            '号码': [str(i) for i in range(0, 10)]
        }
        
        position = '未知位置'
        play_method = '未知玩法'
        clean_content = content
        
        # 识别位置
        for pos_name, keywords in ssc_positions.items():
            for keyword in keywords:
                if keyword in content:
                    position = pos_name
                    clean_content = clean_content.replace(keyword, '').strip()
                    break
            if position != '未知位置':
                break
        
        # 识别玩法
        for method_name, keywords in play_methods.items():
            for keyword in keywords:
                if keyword in clean_content:
                    play_method = method_name
                    break
            if play_method != '未知玩法':
                break
        
        return play_method, position, clean_content

    @staticmethod
    def _parse_3d_content(content):
        """解析3D内容"""
        # 3D位置识别
        three_d_positions = {
            '百位': ['百位'],
            '十位': ['十位'],
            '个位': ['个位']
        }
        
        # 玩法类型识别
        play_methods = {
            '大小': ['大', '小'],
            '单双': ['单', '双'],
            '号码': [str(i) for i in range(0, 10)]
        }
        
        position = '未知位置'
        play_method = '未知玩法'
        clean_content = content
        
        # 识别位置
        for pos_name, keywords in three_d_positions.items():
            for keyword in keywords:
                if keyword in content:
                    position = pos_name
                    clean_content = clean_content.replace(keyword, '').strip()
                    break
            if position != '未知位置':
                break
        
        # 识别玩法
        for method_name, keywords in play_methods.items():
            for keyword in keywords:
                if keyword in clean_content:
                    play_method = method_name
                    break
            if play_method != '未知玩法':
                break
        
        return play_method, position, clean_content

# ==================== 数据分析类 ====================
class DataAnalyzer:
    def __init__(self):
        self.cache = {}
        self.content_parser = ContentParser()  # 添加统一解析器

    @lru_cache(maxsize=1000)
    def cached_extract_numbers(self, content, min_num=0, max_num=49, is_pk10=False):
        """带缓存的号码提取"""
        return self.extract_numbers_from_content(content, min_num, max_num, is_pk10)
    
    @lru_cache(maxsize=500)
    def cached_extract_amount(self, amount_text):
        """带缓存的金额提取"""
        return DataProcessor.enhanced_extract_amount(amount_text)
    
    @lru_cache(maxsize=10000)
    def extract_numbers_cached(self, content, min_num, max_num, is_pk10=False):
        """带缓存的号码提取函数"""
        return self.extract_numbers_from_content(content, min_num, max_num, is_pk10)
    
    def extract_numbers_from_content(self, content, min_num=0, max_num=49, is_pk10=False):
        """从内容中提取数字 - 增强逗号分隔多号码处理"""
        numbers = []
        content_str = str(content)
        
        try:
            # 特殊处理三军格式：1,2,3,4,5,6
            if re.match(r'^(\d,)*\d$', content_str.strip()):
                numbers = [int(x.strip()) for x in content_str.split(',') if x.strip().isdigit()]
                # 过滤范围
                numbers = [num for num in numbers if min_num <= num <= max_num]
                return list(set(numbers))
            
            # 处理常规逗号分隔的数字格式：01,04,05
            if ',' in content_str:
                # 提取所有数字部分
                parts = [p.strip() for p in content_str.split(',')]
                for part in parts:
                    # 检查是否是纯数字
                    if part.isdigit():
                        num = int(part)
                        if min_num <= num <= max_num:
                            numbers.append(num)
                    # 检查是否包含非数字字符（如"冠军-01"）
                    elif re.search(r'\d{1,2}', part):
                        # 提取数字部分
                        num_matches = re.findall(r'\d{1,2}', part)
                        for match in num_matches:
                            num = int(match)
                            if min_num <= num <= max_num:
                                numbers.append(num)
            
            if is_pk10:
                # PK拾/赛车特殊处理：过滤掉"第X名"等玩法描述
                content_str = re.sub(r'第\d+名-?', '', content_str)
            
            # 常规数字提取（处理单个数字）
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
        """增强版斗牛类型提取 - 处理投注内容中的金额信息"""
        content_str = str(content).strip()
        bull_types = []
        
        # 处理金额信息：移除"投注："之后的部分
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 移除"斗牛-"前缀
        if content_str.startswith('斗牛-'):
            content_str = content_str[3:]  # 移除"斗牛-"
        elif content_str.startswith('鬥牛-'):
            content_str = content_str[3:]  # 移除"鬥牛-"
        
        # 如果内容为空，返回空列表
        if not content_str:
            return []
        
        # 定义斗牛类型映射
        type_mapping = {
            '无牛': ['无牛', '無牛', '无', '0牛', '牛0'],
            '牛一': ['牛一', '牛1', '一牛', '1牛'],
            '牛二': ['牛二', '牛2', '二牛', '2牛'],
            '牛三': ['牛三', '牛3', '三牛', '3牛'],
            '牛四': ['牛四', '牛4', '四牛', '4牛'],
            '牛五': ['牛五', '牛5', '五牛', '5牛'],
            '牛六': ['牛六', '牛6', '六牛', '6牛'],
            '牛七': ['牛七', '牛7', '七牛', '7牛'],
            '牛八': ['牛八', '牛8', '八牛', '8牛'],
            '牛九': ['牛九', '牛9', '九牛', '9牛'],
            '牛牛': ['牛牛', '牛10', '十牛', '10牛', '牛十']
        }
        
        # 按逗号分割
        parts = [p.strip() for p in content_str.split(',')]
        
        for part in parts:
            if not part:
                continue
                
            found = False
            for standard_type, variants in type_mapping.items():
                for variant in variants:
                    if variant == part:  # 完全匹配
                        bull_types.append(standard_type)
                        found = True
                        break
                if found:
                    break
            
            # 如果还是没有匹配到，尝试直接匹配（可能是标准类型）
            if not found:
                for standard_type in type_mapping.keys():
                    if standard_type == part:
                        bull_types.append(standard_type)
                        break
        
        return list(set(bull_types))

    def extract_lhc_tema_numbers_and_sides(self, content):
        """从六合彩特码内容中同时提取号码和两面投注 - 增强版"""
        content_str = str(content)
        result = {
            'numbers': set(),
            'has_big': False,
            'has_small': False,
            'has_single': False,
            'has_double': False
        }
        
        # 解析玩法-投注内容格式
        if '-' in content_str:
            parts = content_str.split('-', 1)
            bet_content = parts[1].strip()
        else:
            bet_content = content_str
        
        # 提取号码 (1-49)
        numbers = self.extract_numbers_from_content(bet_content, 1, 49)
        result['numbers'].update(numbers)
        
        # 提取大小单双投注
        content_lower = bet_content.lower()
        
        # 检查特码相关的大小单双投注
        tema_side_keywords = [
            ('大', 'has_big'),
            ('小', 'has_small'),
            ('单', 'has_single'),
            ('双', 'has_double')
        ]
        
        for keyword, field in tema_side_keywords:
            if f'特码{keyword}' in content_lower or f'特码-{keyword}' in content_lower or f'特-{keyword}' in content_lower:
                result[field] = True
        
        # 如果以上都没匹配到，检查普通大小单双（特码玩法中可能直接写"大"、"小"等）
        if not any([result['has_big'], result['has_small'], result['has_single'], result['has_double']]):
            size_parity = self.extract_size_parity_from_content(bet_content)
            for bet in size_parity:
                if bet == '大':
                    result['has_big'] = True
                elif bet == '小':
                    result['has_small'] = True
                elif bet == '单':
                    result['has_single'] = True
                elif bet == '双':
                    result['has_double'] = True
        
        return result
    
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
        """解析PK10号码类玩法内容 - 增强逗号分隔多号码支持"""
        content_str = str(content)
        numbers_by_position = defaultdict(list)
        
        # 首先尝试竖线分隔格式
        if '|' in content_str and any(char.isdigit() or char == '_' or char == ',' for char in content_str):
            vertical_result = ContentParser.parse_pk10_vertical_format(content_str)
            if any(vertical_result.values()):
                return vertical_result
        
        # 处理"位置-号码1,号码2,号码3"格式
        if '-' in content_str and re.search(r'\d{2}', content_str):
            # 先按逗号分割，再处理每个部分
            parts = [part.strip() for part in content_str.split(',')]
            current_position = None
            
            for part in parts:
                if '-' in part:
                    try:
                        position_part, numbers_str = part.split('-', 1)
                        current_position = position_part.strip()
                        numbers_str = numbers_str.strip()
                        
                        # 标准化位置
                        position = self._normalize_pk10_position(current_position)
                        if not position:
                            continue
                        
                        # 提取数字（可能是单个或多个逗号分隔）
                        numbers = []
                        if ',' in numbers_str:
                            # 逗号分隔的多个数字
                            num_parts = numbers_str.split(',')
                            for num_part in num_parts:
                                if num_part.strip().isdigit():
                                    numbers.append(int(num_part.strip()))
                        else:
                            # 单个数字
                            if numbers_str.isdigit():
                                numbers.append(int(numbers_str))
                        
                        # 添加到对应位置
                        numbers_by_position[position].extend(numbers)
                        
                    except ValueError:
                        continue
                elif current_position:
                    # 属于当前位置的投注项（数字部分）
                    position = self._normalize_pk10_position(current_position)
                    if position and part.strip().isdigit():
                        numbers_by_position[position].append(int(part.strip()))
        
        # 处理竖线分隔的格式：01,02,03,04,05|07,08,06,09,10|...
        elif '|' in content_str and re.search(r'\d{2}', content_str):
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
    
    def _normalize_pk10_position(self, position):
        """增强的PK10位置标准化 - 支持更多格式"""
        position_mapping = {
            # 中文标准格式
            '冠军': '冠军', '第1名': '冠军', '第一名': '冠军', '1': '冠军', '1st': '冠军',
            '前一': '冠军', '冠': '冠军',
            '亚军': '亚军', '第2名': '亚军', '第二名': '亚军', '2': '亚军', '2nd': '亚军',
            '亚': '亚军',
            '季军': '第三名', '第3名': '第三名', '第三名': '第三名', '三名': '第三名', '3': '第三名', '3rd': '第三名',
            '第4名': '第四名', '第四名': '第四名', '四名': '第四名', '4': '第四名', '4th': '第四名',
            '第5名': '第五名', '第五名': '第五名', '五名': '第五名', '5': '第五名', '5th': '第五名',
            '第6名': '第六名', '第六名': '第六名', '六名': '第六名', '6': '第六名', '6th': '第六名',
            '第7名': '第七名', '第七名': '第七名', '七名': '第七名', '7': '第七名', '7th': '第七名',
            '第8名': '第八名', '第八名': '第八名', '八名': '第八名', '8': '第八名', '8th': '第八名',
            '第9名': '第九名', '第九名': '第九名', '九名': '第九名', '9': '第九名', '9th': '第九名',
            '第10名': '第十名', '第十名': '第十名', '十名': '第十名', '10': '第十名', '10th': '第十名'
        }
        
        position = str(position).strip()
        
        # 直接映射
        if position in position_mapping:
            return position_mapping[position]
        
        # 模糊匹配 - 增强逻辑
        for key, value in position_mapping.items():
            if key in position:
                return value
        
        # 处理数字格式
        if position.isdigit():
            num = int(position)
            if 1 <= num <= 10:
                if num == 1:
                    return '冠军'
                elif num == 2:
                    return '亚军'
                elif num == 3:
                    return '第三名'
                elif num == 4:
                    return '第四名'
                elif num == 5:
                    return '第五名'
                elif num == 6:
                    return '第六名'
                elif num == 7:
                    return '第七名'
                elif num == 8:
                    return '第八名'
                elif num == 9:
                    return '第九名'
                elif num == 10:
                    return '第十名'
        
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
        """解析六合彩特殊玩法内容，按照玩法-投注内容格式解析 - 修复版本"""
        content_str = str(content).strip()
        
        # 处理简写形式：特大、特小、特单、特双
        if content_str in ['特大', '特小', '特单', '特双']:
            # 简写形式，返回空字符串（因为投注内容就是玩法本身）
            return ''
        
        # 新的解析逻辑：按照"玩法-投注内容"格式解析
        if '-' in content_str:
            parts = content_str.split('-', 1)  # 只分割第一个"-"
            play_method = parts[0].strip()      # 玩法部分
            bet_content = parts[1].strip()      # 投注内容部分
    
            # 返回投注内容部分，这才是实际的下注内容
            return bet_content
        else:
            # 如果没有"-"，整个内容作为投注内容
            return content_str
    
    def extract_lhc_two_sides_content(self, content):
        """专门提取六合彩两面玩法的各种投注类型 - 修复版本"""
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
        
        # 新增：特大、特小映射到普通大小 - 关键修复
        if '特大' in clean_content or clean_content == '特大':
            result['normal_size'].add('大')
        if '特小' in clean_content or clean_content == '特小':
            result['normal_size'].add('小')
    
        # 波色检测
        if '红波' in clean_content and '红波-' not in content_str:
            result['wave'].add('红波')
        if '蓝波' in clean_content and '蓝波-' not in content_str:
            result['wave'].add('蓝波')
        if '绿波' in clean_content and '绿波-' not in content_str:
            result['wave'].add('绿波')
    
        # 普通大小检测（排除特码大小）
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
    
        # 家禽野兽检测
        if '特家肖' in clean_content or '家肖' in clean_content:
            result['animal_type'].add('家禽')
        if '特野肖' in clean_content or '野肖' in clean_content:
            result['animal_type'].add('野兽')
        
        # 普通的家禽野兽
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
        """创建玩法分类映射的完整映射 - 精确版本"""
        mapping = {
            # 快三玩法 - 保留原有的
            '和值': '和值',
            '和值_大小单双': '和值',
            '两面': '两面',
            '二不同号': '二不同号',
            '三不同号': '三不同号',
            '独胆': '独胆',
            '点数': '和值',
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
            '尾数_头尾数': '尾数_头尾数',
            '特尾': '特尾',
            '全尾': '全尾',
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
            '连肖': '连肖',
    
            # 连尾玩法映射
            '二连尾': '二连尾',
            '三连尾': '三连尾',
            '四连尾': '四连尾',
            '五连尾': '五连尾',
            '连肖连尾_二连尾': '二连尾',
            '连肖连尾_三连尾': '三连尾',
            '连肖连尾_四连尾': '四连尾',
            '连肖连尾_五连尾': '五连尾',
            '连尾': '连尾',
    
            # 波色相关玩法
            '色波': '色波',
            '七色波': '色波',
            '波色': '色波',
    
            # 半波相关玩法映射
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
            '斗牛': '斗牛',
            '鬥牛': '斗牛',
            '牛牛': '斗牛',
            '无牛': '斗牛',
            '牛一': '斗牛',
            '牛二': '斗牛',
            '牛三': '斗牛',
            '牛四': '斗牛',
            '牛五': '斗牛',
            '牛六': '斗牛',
            '牛七': '斗牛',
            '牛八': '斗牛',
            '牛九': '斗牛',
            '牛10': '斗牛',
            '牛牛-': '斗牛',
            '斗牛玩法': '斗牛',
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
            
            # PK拾/赛车玩法 - 新增精确映射
            '前一': '冠军',
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
            
            # 龙虎独立玩法 - 新增精确映射（关键修复）
            '龙虎_冠军': '龙虎_冠军',
            '龙虎_冠 军': '龙虎_冠军',
            '龙虎_冠　军': '龙虎_冠军',
            '龙虎_冠  军': '龙虎_冠军',
            '龙虎_亚军': '龙虎_亚军', 
            '龙虎_亚 军': '龙虎_亚军',
            '龙虎_亚　军': '龙虎_亚军',
            '龙虎_亚  军': '龙虎_亚军',
            '龙虎_季军': '龙虎_季军',
            '龙虎_季 军': '龙虎_季军',
            '龙虎_季　军': '龙虎_季军',
            '龙虎_季  军': '龙虎_季军',
            '龙虎_第四名': '龙虎_第四名',
            '龙虎_第五名': '龙虎_第五名',
            '龙虎_第六名': '龙虎_第六名',
            '龙虎_第七名': '龙虎_第七名',
            '龙虎_第八名': '龙虎_第八名',
            '龙虎_第九名': '龙虎_第九名',
            '龙虎_第十名': '龙虎_第十名',
            
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
            '特码': '特码',
            
            # 六合彩正码精确映射 - 新增精确映射（关键修复）
            '正码1-6_正码一': '正码一',
            '正码1-6_正码二': '正码二',
            '正码1-6_正码三': '正码三',
            '正码1-6_正码四': '正码四',
            '正码1-6_正码五': '正码五',
            '正码1-6_正码六': '正码六',
            '正码一': '正码一',
            '正码二': '正码二',
            '正码三': '正码三',
            '正码四': '正码四',
            '正码五': '正码五',
            '正码六': '正码六'
        }
        return mapping
    
    def normalize_category(self, category):
        """统一玩法分类名称 - 修复版本"""
        category_str = str(category).strip()
        
        # 首先规范化特殊字符
        import re
        category_normalized = re.sub(r'\s+', ' ', category_str)
        
        # 直接映射（使用标准化后的字符串）
        if category_normalized in self.category_mapping:
            return self.category_mapping[category_normalized]
        
        # 关键词匹配（使用标准化后的字符串）
        for key, value in self.category_mapping.items():
            if key in category_normalized:
                return value
        
        # 特别处理通用的"龙虎"分类
        if category_normalized == '龙虎':
            return '龙虎'  # 保持为通用龙虎
        
        # 使用标准化后的字符串生成小写版本
        category_lower = category_normalized.lower()
        
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
        
        return category_normalized  # 返回标准化后的字符串

# ==================== 分析引擎 ====================
class AnalysisEngine:
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.normalizer = PlayCategoryNormalizer()
        self.seen_records = set()  # 用于记录已检测的记录
        self.enhanced_parser = EnhancedContentParser()

    def parse_play_content_enhanced(self, content, current_category, lottery_type):
        """增强版内容解析 - 返回实际玩法分类和投注内容"""
        content_str = str(content)
        
        # 根据彩种类型定义玩法关键字映射
        play_keywords_mapping = {
            'LHC': {
                # 尾数玩法
                '特尾': '特尾',
                '全尾': '全尾',
                '头尾数': '尾数_头尾数',
                '尾数': '尾数',
                # 正码特玩法
                '正码一特': '正1特',
                '正码二特': '正2特', 
                '正码三特': '正3特',
                '正码四特': '正4特',
                '正码五特': '正5特',
                '正码六特': '正6特',
                # 连肖玩法
                '二连肖': '连肖连尾_二连肖',
                '三连肖': '连肖连尾_三连肖',
                '四连肖': '连肖连尾_四连肖', 
                '五连肖': '连肖连尾_五连肖',
                # 连尾玩法
                '二连尾': '连肖连尾_二连尾',
                '三连尾': '连肖连尾_三连尾',
                '四连尾': '连肖连尾_四连尾',
                '五连尾': '连肖连尾_五连尾'
            },
            'PK10': {
                # 位置信息
                '冠军': '冠军',
                '亚军': '亚军',
                '第三名': '第三名',
                '第四名': '第四名',
                '第五名': '第五名',
                '第六名': '第六名', 
                '第七名': '第七名',
                '第八名': '第八名',
                '第九名': '第九名',
                '第十名': '第十名',
                '前一': '冠军'
            },
            'SSC': {
                # 位置信息
                '第1球': '第1球',
                '第2球': '第2球',
                '第3球': '第3球',
                '第4球': '第4球', 
                '第5球': '第5球',
                '万位': '第1球',
                '千位': '第2球',
                '百位': '第3球',
                '十位': '第4球',
                '个位': '第5球'
            },
            '3D': {
                # 位置信息
                '百位': '百位',
                '十位': '十位',
                '个位': '个位'
            }
        }
        
        # 获取对应彩种的玩法映射
        play_keywords = play_keywords_mapping.get(lottery_type, {})
        
        # 检查内容中是否包含玩法关键字
        detected_play_method = None
        for keyword, play_method in play_keywords.items():
            if keyword in content_str:
                detected_play_method = play_method
                break
        
        # 提取投注内容
        bet_content = content_str
        if '-' in content_str:
            parts = content_str.split('-', 1)
            if len(parts) == 2:
                bet_content = parts[1].strip()
        
        return detected_play_method, bet_content

    def normalize_play_category_from_content(self, content, current_category, lottery_type):
        """基于内容统一标准化玩法分类"""
        detected_play_method, _ = self.parse_play_content_enhanced(content, current_category, lottery_type)
        
        if detected_play_method:
            return detected_play_method
        else:
            return current_category
    
    def _get_record_hash(self, record):
        """生成记录的唯一哈希值"""
        key_parts = [
            record['会员账号'],
            record['彩种'], 
            record['期号'],
            record.get('玩法分类', ''),
            record.get('违规类型', ''),
            record.get('投注项', ''),  # 添加投注项，确保不同的投注项不会被合并
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
    
    def normalize_play_categories(self, df):
        """统一玩法分类 - 增强版，结合玩法和内容"""
        logger.info("正在统一玩法分类...")
        
        if '玩法' in df.columns:
            # 第一步：基于玩法列进行初步分类
            df['玩法分类'] = df['玩法'].apply(self.normalizer.normalize_category)
            
            # 第二步：对于分类不明确的记录，从内容中提取玩法
            vague_categories = ['正码', '正码1-6', '龙虎', '两面', '定位胆']
            
            for idx, row in df.iterrows():
                current_category = str(row['玩法分类'])
                content = str(row['内容'])
                lottery = str(row['彩种'])
                lottery_type = self.identify_lottery_type(lottery)
                
                # 如果当前分类比较模糊，尝试从内容中提取更精确的信息
                if any(vague in current_category for vague in vague_categories):
                    play_method, position, clean_content = self.enhanced_parser.extract_play_method_and_position(
                        content, lottery_type
                    )
                    
                    # 如果从内容中识别到了更精确的信息，更新分类
                    if position != '未知位置':
                        new_category = f"{position}"
                        if play_method != '未知玩法':
                            new_category = f"{position}_{play_method}"
                        
                        df.at[idx, '玩法分类'] = new_category
                        # 保存提取的位置信息供后续分析使用
                        df.at[idx, 'extracted_position'] = position
                        df.at[idx, 'extracted_play_method'] = play_method

        return df
    
    def identify_lottery_type(self, lottery_name):
        """识别彩种类型"""
        lottery_str = str(lottery_name).strip()
        
        for lottery_type, config in LOTTERY_CONFIGS.items():
            for lottery in config['lotteries']:
                if lottery in lottery_str:
                    return lottery_type
        
        lottery_lower = lottery_str.lower()
        
        # 更精确的彩种识别
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
        # 增强3D系列识别
        elif any(word in lottery_lower for word in ['排列三', '排列3', '福彩3d', '3d', '极速3d', '排列', 'p3', 'p三']):
            return '3D'
        
        return None

    def normalize_tail_play_category(self, content, current_category):
        """统一标准化尾数玩法分类"""
        content_str = str(content)
        
        # 玩法关键字优先级（从具体到一般）
        play_keywords = [
            ('特尾', '特尾'),
            ('全尾', '全尾'),
            ('头尾数', '尾数_头尾数'),
            ('尾数', '尾数')
        ]
        
        for keyword, normalized_category in play_keywords:
            if keyword in content_str:
                return normalized_category
        
        # 如果没有匹配关键字，返回原始分类
        return current_category

    # =============== PK10分析方法 ===============
    def analyze_pk10_patterns(self, df, period_amount_dict):
        """分析PK拾/赛车系列投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == 'PK10']
        
        if len(df_target) == 0:
            return results
        
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            # 统一的投注项多位置检测（覆盖号码、大小、单双、龙虎）
            self._analyze_pk10_bet_item_multiple_positions(account, lottery, period, group, results, period_amount)
            
            # 原有的其他检测方法
            self._analyze_pk10_two_sides(account, lottery, period, group, results, period_amount)
            self._analyze_pk10_gyh(account, lottery, period, group, results, period_amount)
            self._analyze_pk10_number_plays(account, lottery, period, group, results, period_amount)
            self._analyze_pk10_independent_plays(account, lottery, period, group, results, period_amount)
            self._analyze_pk10_qianyi_plays(account, lottery, period, group, results, period_amount)
            self._analyze_pk10_dragon_tiger_detailed(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_pk10_two_sides(self, account, lottery, period, group, results, period_amount):

        """分析PK10两面玩法"""
        two_sides_categories = ['两面', '双面']
        
        two_sides_group = group[group['玩法分类'].isin(two_sides_categories)]
        
        position_bets = defaultdict(set)
        
        for _, row in two_sides_group.iterrows():
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
                    '玩法分类': '两面',
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
    
    def _analyze_pk10_gyh(self, account, lottery, period, group, results, period_amount):
        """分析PK10冠亚和玩法"""
        gyh_categories = ['冠亚和', '冠亚和_大小单双', '冠亚和_和值']
        
        gyh_group = group[group['玩法分类'].isin(gyh_categories)]
        
        all_numbers = set()
        all_size_parity = set()
        
        for _, row in gyh_group.iterrows():
            content = str(row['内容'])
            
            # 改进：提取所有数字，不限制范围
            numbers = re.findall(r'\b\d{1,2}\b', content)
            numbers = [int(num) for num in numbers if 1 <= int(num) <= 19]  # 冠亚和范围3-19，但允许提取1-19
            all_numbers.update(numbers)
            
            size_parity = self.data_analyzer.extract_size_parity_from_content(content)
            all_size_parity.update(size_parity)
        
        # 冠亚和多码检测 - 使用所有提取的数字
        if len(all_numbers) >= THRESHOLD_CONFIG['PK10']['gyh_multi_number']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '冠亚和',
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([str(num) for num in sorted(all_numbers)]),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '冠亚和多码')
            }
            self._add_unique_result(results, '冠亚和多码', record)
            return  # 如果检测到多号码，不再检测其他类型
        
        # 原有的矛盾检测逻辑保持不变...
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
                '玩法分类': '冠亚和',
                '矛盾类型': '、'.join(conflicts),
                '投注内容': ', '.join(sorted(all_size_parity)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '冠亚和矛盾')
            }
            self._add_unique_result(results, '冠亚和矛盾', record)
        
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
                '玩法分类': '冠亚和',
                '矛盾类型': '、'.join(conflicts),
                '投注内容': ', '.join(sorted(all_size_parity)),
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '冠亚和矛盾')
            }
            self._add_unique_result(results, '冠亚和矛盾', record)
    
    def _analyze_pk10_number_plays(self, account, lottery, period, group, results, period_amount):

        """分析PK10号码类玩法 - 增强位置判断"""
        number_categories = [
            '1-5名', '6-10名', '冠军', '前一', '亚军', '第三名', '第四名', '第五名',
            '第六名', '第七名', '第八名', '第九名', '第十名', '定位胆',
            '定位胆_第1~5名', '定位胆_第6~10名'
        ]
        
        number_group = group[group['玩法分类'].isin(number_categories)]
        
        all_numbers_by_position = defaultdict(set)
        
        for _, row in number_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 修复：优先从extracted_position字段获取位置（如果存在）
            if 'extracted_position' in row and pd.notna(row['extracted_position']):
                inferred_position = row['extracted_position']
            else:
                # 如果extracted_position不存在，则从玩法分类中提取
                inferred_position = self._extract_exact_pk10_position_from_category(category)
                if inferred_position == '未知位置':
                    # 最后从内容中提取
                    play_method, inferred_position, clean_content = self.enhanced_parser.extract_play_method_and_position(content, 'PK10')
            
            # 使用统一解析器
            bets_by_position = ContentParser.parse_pk10_content(content)
            
            for position, bets in bets_by_position.items():
                # 如果解析出的位置是"未知位置"，使用从玩法分类推断的位置
                if position == '未知位置' and inferred_position:
                    position = inferred_position
                
                # 提取每个位置的号码
                for bet in bets:
                    numbers = self.data_analyzer.extract_numbers_from_content(bet, 1, 10, is_pk10=True)
                    all_numbers_by_position[position].update(numbers)

        # 只保留原有的检查每个位置的超码逻辑
        for position, numbers in all_numbers_by_position.items():
            if len(numbers) >= THRESHOLD_CONFIG['PK10']['multi_number']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': f'{position}多码',
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}: {', '.join([f'{num:02d}' for num in sorted(numbers)])}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, f'{position}多码')
                }
                self._add_unique_result(results, '超码', record)
    
    def _infer_position_from_category(self, category):
        """从玩法分类推断位置"""
        category_str = str(category).strip()
        
        position_mapping = {
            '冠军': ['冠军', '前一', '第1名', '第一名'],
            '亚军': ['亚军', '第2名', '第二名'],
            '第三名': ['第三名', '季军', '第3名'],
            '第四名': ['第四名', '第4名'],
            '第五名': ['第五名', '第5名'],
            '第六名': ['第六名', '第6名'],
            '第七名': ['第七名', '第7名'],
            '第八名': ['第八名', '第8名'],
            '第九名': ['第九名', '第9名'],
            '第十名': ['第十名', '第10名'],
            '1-5名': ['1-5名', '定位胆_第1~5名'],
            '6-10名': ['6-10名', '定位胆_第6~10名']
        }
        
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                if keyword in category_str:
                    return position
        
        return None
    
    def _analyze_pk10_independent_plays(self, account, lottery, period, group, results, period_amount):
        """分析PK10独立玩法（大小单双）- 排除龙虎，避免重复检测"""
        # 只保留大小和单双的独立玩法，排除龙虎（因为龙虎有专门的检测）
        independent_categories = [
            '大小_冠军', '大小_亚军', '大小_季军',
            '单双_冠军', '单双_亚军', '单双_季军'
            # 移除龙虎相关的独立玩法，因为龙虎有专门的检测
        ]
        
        independent_group = group[group['玩法分类'].isin(independent_categories)]
        
        position_bets = defaultdict(set)
        
        for _, row in independent_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 确定位置（前一就是冠军）
            if '冠军' in category or '前一' in category:
                position = '冠军'
            elif '亚军' in category:
                position = '亚军'
            elif '季军' in category:
                position = '季军'
            else:
                continue
            
            # 只提取大小单双，不提取龙虎
            bets = self.data_analyzer.extract_size_parity_from_content(content)
            # 过滤掉龙虎相关的投注
            bets = [bet for bet in bets if bet not in ['龙', '虎']]
            
            if bets:  # 只有当有大小单双投注时才记录
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
                    '玩法分类': '独立玩法',
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '独立玩法矛盾')
                }
                self._add_unique_result(results, '独立玩法矛盾', record)
    
    def _analyze_pk10_qianyi_plays(self, account, lottery, period, group, results, period_amount):
        """分析PK10前一玩法"""
        qianyi_categories = ['前一']
        
        qianyi_group = group[group['玩法分类'].isin(qianyi_categories)]
        
        for _, row in qianyi_group.iterrows():
            content = str(row['内容'])
            
            # 提取号码
            numbers = self.data_analyzer.extract_numbers_from_content(
                content,
                LOTTERY_CONFIGS['PK10']['min_number'],
                LOTTERY_CONFIGS['PK10']['max_number']
            )
            
            # 前一多码检测（前一就是冠军，所以使用冠军的阈值）
            if len(numbers) >= THRESHOLD_CONFIG['PK10']['multi_number']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '前一',
                    '位置': '冠军',  # 显示为冠军位置
                    '号码数量': len(numbers),
                    '投注内容': ', '.join([f'{num:02d}' for num in sorted(numbers)]),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '超码')
                }
                self._add_unique_result(results, '超码', record)
    
    def _extract_position_from_dragon_tiger_category(self, category):
        """从龙虎玩法分类中直接提取位置 - 清理版本"""
        category_str = str(category).strip()
        
        import re
        category_normalized = re.sub(r'\s+', ' ', category_str)
        
        position_mapping = {
            '龙虎_冠军': '冠军', '龙虎_冠 军': '冠军', '龙虎_冠　军': '冠军', '龙虎_冠  军': '冠军',
            '冠军': '冠军', '龙虎_前一': '冠军',
            '龙虎_亚军': '亚军', '龙虎_亚 军': '亚军', '龙虎_亚　军': '亚军', '龙虎_亚  军': '亚军',
            '亚军': '亚军',
            '龙虎_季军': '季军', '龙虎_季 军': '季军', '龙虎_季　军': '季军', '龙虎_季  军': '季军',
            '季军': '季军', '龙虎_第三名': '季军',
            '龙虎_第四名': '第四名', '龙虎_第五名': '第五名', '龙虎_第六名': '第六名',
            '龙虎_第七名': '第七名', '龙虎_第八名': '第八名', '龙虎_第九名': '第九名', '龙虎_第十名': '第十名'
        }
        
        if category_normalized in position_mapping:
            return position_mapping[category_normalized]
        
        for key, value in position_mapping.items():
            if key in category_normalized:
                return value
        
        if '冠军' in category_normalized or '前一' in category_normalized:
            return '冠军'
        elif '亚军' in category_normalized:
            return '亚军'
        elif '季军' in category_normalized or '第三名' in category_normalized:
            return '季军'
        elif '第四名' in category_normalized:
            return '第四名'
        elif '第五名' in category_normalized:
            return '第五名'
        elif '第六名' in category_normalized:
            return '第六名'
        elif '第七名' in category_normalized:
            return '第七名'
        elif '第八名' in category_normalized:
            return '第八名'
        elif '第九名' in category_normalized:
            return '第九名'
        elif '第十名' in category_normalized:
            return '第十名'
        
        return '未知位置'

    def _analyze_pk10_dragon_tiger_detailed(self, account, lottery, period, group, results, period_amount):
        """PK10龙虎详细检测 - 清理版本"""
        dragon_tiger_categories = [
            '龙虎_冠军', '龙虎_亚军', '龙虎_季军', '龙虎', '龙虎_第四名', '龙虎_第五名', 
            '龙虎_第六名', '龙虎_第七名', '龙虎_第八名', '龙虎_第九名', '龙虎_第十名',
            '龙虎_冠 军', '龙虎_亚 军', '龙虎_季 军',
            '龙虎_冠　军', '龙虎_亚　军', '龙虎_季　军',
            '龙虎_冠  军', '龙虎_亚  军', '龙虎_季  军',
            '龙虎_前一'
        ]
        
        dragon_tiger_group = group[group['玩法分类'].isin(dragon_tiger_categories)]
        
        position_bets = defaultdict(set)
        
        for _, row in dragon_tiger_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            position = self._extract_position_from_dragon_tiger_category(category)
            dragon_tiger = self.data_analyzer.extract_dragon_tiger_from_content(content)
            
            if dragon_tiger:
                position_bets[position].update(dragon_tiger)
        
        for position, bets in position_bets.items():
            if '龙' in bets and '虎' in bets:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '龙虎',
                    '位置': position,
                    '矛盾类型': '龙虎矛盾',
                    '投注内容': f"{position}-{','.join(sorted(bets))}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '龙虎矛盾'}, '龙虎矛盾')
                }
                self._add_unique_result(results, '龙虎矛盾', record)

    def _analyze_pk10_bet_item_multiple_positions(self, account, lottery, period, group, results, period_amount):
        """统一的多位置相同投注检测 - 修复版本：准确识别实际投注位置"""
        
        # 收集所有位置的投注项
        position_bet_items = defaultdict(set)
        
        for _, row in group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 修复：对于"1-5名"和"6-10名"，需要解析内容中的实际位置和多个号码
            if category in ['1-5名', '1~5名', '1-5', '1~5', '1-5名定位胆']:
                # 解析1-5名投注内容（使用增强版）
                bets_by_position = self._parse_1_5_content_detailed_enhanced(content)
                
                for position, bet_items in bets_by_position.items():
                    if bet_items:
                        # 标准化位置名称
                        normalized_position = self.data_analyzer._normalize_pk10_position(position)
                        if normalized_position:
                            position_bet_items[normalized_position].update(bet_items)
                            
            elif category in ['6-10名', '6~10名', '6-10', '6~10', '6-10名定位胆']:
                # 解析6-10名投注内容（使用增强版）
                bets_by_position = self._parse_6_10_content_detailed_enhanced(content)
                
                for position, bet_items in bets_by_position.items():
                    if bet_items:
                        # 标准化位置名称
                        normalized_position = self.data_analyzer._normalize_pk10_position(position)
                        if normalized_position:
                            position_bet_items[normalized_position].update(bet_items)
                            
            else:
                # 解析其他投注内容，提取投注项
                bet_items_by_position = self._extract_all_bet_items_from_content(content)
                
                for position, bet_items in bet_items_by_position.items():
                    if bet_items:
                        # 标准化位置名称
                        normalized_position = self.data_analyzer._normalize_pk10_position(position)
                        if normalized_position:
                            position_bet_items[normalized_position].update(bet_items)
        
        # 统计每个投注项出现的位置
        bet_item_to_positions = defaultdict(set)
        for position, bet_items in position_bet_items.items():
            for bet_item in bet_items:
                bet_item_to_positions[bet_item].add(position)
        
        # 检测阈值配置
        MULTI_POSITION_THRESHOLD = THRESHOLD_CONFIG['PK10']['same_number_multi_position']
        
        # 检查每个投注项
        for bet_item, positions in bet_item_to_positions.items():
            position_count = len(positions)
            
            if position_count >= MULTI_POSITION_THRESHOLD:
                # 确定投注项类型
                if bet_item.isdigit():
                    item_type = '号码'
                elif bet_item in ['大', '小']:
                    item_type = '大小'
                elif bet_item in ['单', '双']:
                    item_type = '单双'
                elif bet_item in ['龙', '虎']:
                    item_type = '龙虎'
                else:
                    item_type = '投注项'
                
                # 生成投注内容描述
                if position_count == 10:
                    result_key = '十个位置相同投注'
                    if item_type == '号码':
                        play_category = f'十个位置相同号码投注'
                        details = f'号码{bet_item}在十个位置投注'
                        bet_content = f'号码{bet_item}'
                    elif item_type == '大小':
                        play_category = f'十个位置相同大小投注'
                        details = f'大小{bet_item}在十个位置投注'
                        bet_content = f'{bet_item}'
                    elif item_type == '单双':
                        play_category = f'十个位置相同单双投注'
                        details = f'单双{bet_item}在十个位置投注'
                        bet_content = f'{bet_item}'
                    else:
                        play_category = f'十个位置相同{item_type}投注'
                        details = f'{item_type}{bet_item}在十个位置投注'
                        bet_content = f'{bet_item}'
                else:
                    result_key = '多位置相同投注'
                    if item_type == '号码':
                        play_category = f'{position_count}个位置相同号码投注'
                        details = f'号码{bet_item}在{position_count}个位置投注'
                        bet_content = f'号码{bet_item}'
                    elif item_type == '大小':
                        play_category = f'{position_count}个位置相同大小投注'
                        details = f'大小{bet_item}在{position_count}个位置投注'
                        bet_content = f'{bet_item}'
                    elif item_type == '单双':
                        play_category = f'{position_count}个位置相同单双投注'
                        details = f'单双{bet_item}在{position_count}个位置投注'
                        bet_content = f'{bet_item}'
                    else:
                        play_category = f'{position_count}个位置相同{item_type}投注'
                        details = f'{item_type}{bet_item}在{position_count}个位置投注'
                        bet_content = f'{bet_item}'
                
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': play_category,
                    '违规类型': result_key,
                    '投注项': bet_item,
                    '投注类型': item_type,
                    '位置数量': position_count,
                    '出现位置': '、'.join(sorted(positions)),
                    '详细信息': details,
                    '投注内容': bet_content,
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'位置数量': position_count}, result_key)
                }
                
                self._add_unique_result(results, result_key, record)
    
    def _parse_1_5_content_detailed(self, content):
        """详细解析1-5名投注内容 - 增强逗号分隔多号码支持"""
        content_str = str(content).strip()
        bets_by_position = defaultdict(set)
        
        if not content_str:
            return bets_by_position
        
        # 移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 使用统一的逗号分隔多号码解析
        parts = [p.strip() for p in content_str.split(',')]
        current_position = None
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否包含位置信息
            if '-' in part:
                try:
                    position_part, bet_part = part.split('-', 1)
                    current_position = position_part.strip()
                    bet_part = bet_part.strip()
                    
                    # 标准化位置
                    normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                    if not normalized_position:
                        continue
                    
                    # 提取投注项
                    self._extract_bet_items_from_part_enhanced(bet_part, bets_by_position, normalized_position)
                    
                except ValueError:
                    continue
            elif current_position:
                # 属于当前位置的投注项
                normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                if normalized_position:
                    self._extract_bet_items_from_part_enhanced(part, bets_by_position, normalized_position)
        
        return bets_by_position
    
    def _parse_6_10_content_detailed(self, content):
        """详细解析6-10名投注内容 - 增强逗号分隔多号码支持"""
        content_str = str(content).strip()
        bets_by_position = defaultdict(set)
        
        if not content_str:
            return bets_by_position
        
        # 移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 使用统一的逗号分隔多号码解析
        parts = [p.strip() for p in content_str.split(',')]
        current_position = None
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否包含位置信息
            if '-' in part:
                try:
                    position_part, bet_part = part.split('-', 1)
                    current_position = position_part.strip()
                    bet_part = bet_part.strip()
                    
                    # 标准化位置
                    normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                    if not normalized_position:
                        continue
                    
                    # 提取投注项
                    self._extract_bet_items_from_part_enhanced(bet_part, bets_by_position, normalized_position)
                    
                except ValueError:
                    continue
            elif current_position:
                # 属于当前位置的投注项
                normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                if normalized_position:
                    self._extract_bet_items_from_part_enhanced(part, bets_by_position, normalized_position)
        
        return bets_by_position

    def _extract_bet_items_from_part_enhanced(self, bet_part, bets_by_position, position):
        """从投注部分提取投注项（增强多号码支持）"""
        bet_part = bet_part.strip()
        
        if not bet_part:
            return
        
        # 处理逗号分隔的多个投注项
        if ',' in bet_part:
            sub_parts = [p.strip() for p in bet_part.split(',')]
            for sub_part in sub_parts:
                standardized_item = self._standardize_bet_item(sub_part)
                if standardized_item:
                    bets_by_position[position].add(standardized_item)
        else:
            # 单个投注项
            standardized_item = self._standardize_bet_item(bet_part)
            if standardized_item:
                bets_by_position[position].add(standardized_item)

    def _parse_1_5_content_detailed_enhanced(self, content):
        """增强解析1-5名投注内容 - 完整支持逗号分隔多号码"""
        content_str = str(content).strip()
        bets_by_position = defaultdict(set)
        
        if not content_str:
            return bets_by_position
        
        # 移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 使用统一的逗号分隔多号码解析
        parts = [p.strip() for p in content_str.split(',')]
        current_position = None
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否包含位置信息
            if '-' in part:
                try:
                    position_part, bet_part = part.split('-', 1)
                    current_position = position_part.strip()
                    bet_part = bet_part.strip()
                    
                    # 标准化位置
                    normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                    if not normalized_position:
                        continue
                    
                    # 提取投注项
                    self._extract_bet_items_from_part_enhanced(bet_part, bets_by_position, normalized_position)
                    
                except ValueError:
                    continue
            elif current_position:
                # 属于当前位置的投注项
                normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                if normalized_position:
                    self._extract_bet_items_from_part_enhanced(part, bets_by_position, normalized_position)
        
        return bets_by_position
    
    def _parse_6_10_content_detailed_enhanced(self, content):
        """增强解析6-10名投注内容 - 完整支持逗号分隔多号码"""
        content_str = str(content).strip()
        bets_by_position = defaultdict(set)
        
        if not content_str:
            return bets_by_position
        
        # 移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 使用统一的逗号分隔多号码解析
        parts = [p.strip() for p in content_str.split(',')]
        current_position = None
        
        for part in parts:
            if not part:
                continue
            
            # 检查是否包含位置信息
            if '-' in part:
                try:
                    position_part, bet_part = part.split('-', 1)
                    current_position = position_part.strip()
                    bet_part = bet_part.strip()
                    
                    # 标准化位置
                    normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                    if not normalized_position:
                        continue
                    
                    # 提取投注项
                    self._extract_bet_items_from_part_enhanced(bet_part, bets_by_position, normalized_position)
                    
                except ValueError:
                    continue
            elif current_position:
                # 属于当前位置的投注项
                normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                if normalized_position:
                    self._extract_bet_items_from_part_enhanced(part, bets_by_position, normalized_position)
        
        return bets_by_position

    def _extract_all_bet_items_from_content(self, content):
        """从内容中提取所有类型的投注项（号码、大小、单双、龙虎）- 增强逗号分隔多号码支持"""
        content_str = str(content).strip()
        bets_by_position = defaultdict(set)
        
        if not content_str:
            return bets_by_position
        
        # 先移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 检查是否为竖线格式
        if '|' in content_str:
            parts = content_str.split('|')
            positions = ['冠军', '亚军', '第三名', '第四名', '第五名', 
                        '第六名', '第七名', '第八名', '第九名', '第十名']
            
            for i, part in enumerate(parts):
                if i < len(positions):
                    position = positions[i]
                    part_clean = part.strip()
                    
                    # 跳过空位或下划线
                    if not part_clean or part_clean == '_' or part_clean == '':
                        continue
                    
                    # 提取数字（包括逗号分隔的多个数字）
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
                    
                    # 记录数字投注项
                    for num in numbers:
                        standardized_item = self._standardize_bet_item(str(num))
                        if standardized_item:
                            bets_by_position[position].add(standardized_item)
        
        # 检查是否为"位置-号码"格式（包含逗号分隔的多个号码）
        elif '-' in content_str and not any(x in content_str for x in ['投注：', '抵用：', '中奖：']):
            # 修复：处理多个逗号分隔的号码
            parts = content_str.split(',')
            current_position = None
            
            for part in parts:
                part_clean = part.strip()
                
                if '-' in part_clean:
                    try:
                        position_part, bet_part = part_clean.split('-', 1)
                        current_position = position_part.strip()
                        bet_part = bet_part.strip()
                        
                        # 标准化位置
                        normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                        if not normalized_position:
                            continue
                        
                        # 处理投注部分（可能包含逗号分隔的多个数字）
                        self._extract_numbers_from_bet_part(bet_part, bets_by_position, normalized_position)
                        
                    except ValueError:
                        continue
                elif current_position:
                    # 没有位置信息，但有当前上下文位置，则视为当前位置的投注项
                    normalized_position = self.data_analyzer._normalize_pk10_position(current_position)
                    if normalized_position:
                        # 处理可能是数字的情况
                        self._extract_numbers_from_bet_part(part_clean, bets_by_position, normalized_position)
        
        return bets_by_position
    
    def _extract_numbers_from_bet_part(self, bet_part, bets_by_position, position):
        """从投注部分提取数字并添加到指定位置"""
        bet_part = bet_part.strip()
        
        if not bet_part:
            return
        
        # 处理逗号分隔的多个数字
        if ',' in bet_part:
            number_strs = bet_part.split(',')
            for num_str in number_strs:
                num_clean = num_str.strip()
                standardized_item = self._standardize_bet_item(num_clean)
                if standardized_item:
                    bets_by_position[position].add(standardized_item)
        else:
            # 单个数字
            standardized_item = self._standardize_bet_item(bet_part)
            if standardized_item:
                bets_by_position[position].add(standardized_item)

    def _extract_bet_items_from_1_5_content(self, content):
        """从1-5名或6-10名的投注内容中提取投注项"""
        content_str = str(content).strip()
        bet_items = set()
        
        if not content_str:
            return bet_items
        
        # 先移除金额信息
        if '投注：' in content_str:
            content_str = content_str.split('投注：')[0].strip()
        
        # 处理格式：第三名-01,04,05,第五名-01,04,05,亚军-01,04,05,第四名-01,04,05,冠军-01,04,05
        parts = [part.strip() for part in content_str.split(',')]
        
        for part in parts:
            if not part:
                continue
                
            # 如果包含"-"，说明是位置-投注项格式
            if '-' in part:
                try:
                    position_part, bet_part = part.split('-', 1)
                    position_part = position_part.strip()
                    bet_part = bet_part.strip()
                    
                    # 检查投注部分是否包含逗号（多个投注项）
                    if ',' in bet_part:
                        bet_subparts = [b.strip() for b in bet_part.split(',')]
                        for bet_item in bet_subparts:
                            standardized_item = self._standardize_bet_item(bet_item)
                            if standardized_item:
                                bet_items.add(standardized_item)
                    else:
                        standardized_item = self._standardize_bet_item(bet_part)
                        if standardized_item:
                            bet_items.add(standardized_item)
                            
                except ValueError:
                    continue
            else:
                # 可能是单独的投注项
                standardized_item = self._standardize_bet_item(part)
                if standardized_item:
                    bet_items.add(standardized_item)
        
        return bet_items

    def _standardize_bet_item(self, bet_item):
        """标准化投注项"""
        bet_item = str(bet_item).strip()
        
        if not bet_item:
            return None
        
        # 移除可能的空格和非数字字符
        bet_item = re.sub(r'\s+', '', bet_item)
        
        # 处理两位数字（如"01"转为"1"）
        if re.match(r'^\d{2}$', bet_item):
            return str(int(bet_item))  # 去掉前导0
        
        # 返回大小单双龙虎
        if bet_item in ['大', '小', '单', '双', '龙', '虎']:
            return bet_item
        
        # 处理数字（如"5"）
        if bet_item.isdigit():
            return bet_item
        
        return None
    
    def _extract_bet_items_from_part(self, bet_part, bet_items_by_position, position):
        """从投注部分提取投注项并存储"""
        # 标准化位置
        normalized_position = self.data_analyzer._normalize_pk10_position(position)
        if not normalized_position:
            return
        
        # 检查是否是数字
        if bet_part.isdigit():
            bet_items_by_position[normalized_position].add(bet_part)
        # 检查是否是大小单双龙虎
        elif bet_part in ['大', '小', '单', '双', '龙', '虎']:
            bet_items_by_position[normalized_position].add(bet_part)
        # 处理"01"这样的两位数字
        elif re.match(r'^\d{2}$', bet_part):
            bet_items_by_position[normalized_position].add(str(int(bet_part)))  # 去掉前导0

    def _analyze_pk10_all_positions_bet(self, account, lottery, period, group, results):
        """检测PK10十个位置投注完全相同内容的情况 - 严格版本"""
        
        # 定义十个标准位置
        standard_positions = ['冠军', '亚军', '第三名', '第四名', '第五名', 
                             '第六名', '第七名', '第八名', '第九名', '第十名']
        
        # 收集所有位置的具体投注内容
        position_bets = defaultdict(set)
        
        # 从各种玩法中收集投注信息
        self._collect_detailed_position_bets(account, lottery, period, group, position_bets)
        
        # 检查十个位置是否都有投注
        positions_with_bets = [pos for pos in standard_positions if position_bets.get(pos)]
        
        if len(positions_with_bets) < 10:
            return  # 不是十个位置都有投注
        
        # 检查十个位置的投注内容是否完全相同
        all_bet_sets = []
        for position in standard_positions:
            bets = position_bets.get(position, set())
            # 使用frozenset便于比较，因为set本身不可哈希
            all_bet_sets.append(frozenset(bets))
        
        # 如果十个位置的投注内容集合都相同
        if len(set(all_bet_sets)) == 1:
            common_bets = all_bet_sets[0]  # 获取共同的投注内容
            
            if common_bets:  # 确保不是空集合
                # 分析具体的投注类型
                bet_description = self._analyze_identical_bets_detailed(common_bets)
                
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '十个位置相同投注',
                    '投注位置数': 10,
                    '投注类型': bet_description,
                    '投注内容': f"十个位置相同投注: {bet_description}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'投注位置数': 10}, '十个位置相同投注')
                }
                self._add_unique_result(results, '十个位置相同投注', record)
    
    def _collect_detailed_position_bets(self, account, lottery, period, group, position_bets):
        """详细收集位置投注信息 - 专门用于十个位置相同投注检测"""
        
        # 1. 从两面玩法收集
        two_sides_categories = ['两面', '双面']
        two_sides_group = group[group['玩法分类'].isin(two_sides_categories)]
        
        for _, row in two_sides_group.iterrows():
            content = str(row['内容'])
            self._extract_detailed_bets_from_content(content, position_bets)
        
        # 2. 从独立玩法收集（大小单双）
        independent_categories = [
            '大小_冠军', '大小_亚军', '大小_季军', '大小_第四名', '大小_第五名',
            '大小_第六名', '大小_第七名', '大小_第八名', '大小_第九名', '大小_第十名',
            '单双_冠军', '单双_亚军', '单双_季军', '单双_第四名', '单双_第五名',
            '单双_第六名', '单双_第七名', '单双_第八名', '单双_第九名', '单双_第十名'
        ]
        
        independent_group = group[group['玩法分类'].isin(independent_categories)]
        
        for _, row in independent_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 确定位置
            position = self._extract_position_from_independent_category(category)
            if position:
                # 提取具体的投注内容
                bets = self.data_analyzer.extract_size_parity_from_content(content)
                if bets:
                    position_bets[position].update(bets)
        
        # 3. 从号码类玩法收集
        number_categories = [
            '1-5名', '6-10名', '冠军', '前一', '亚军', '第三名', '第四名', '第五名',
            '第六名', '第七名', '第八名', '第九名', '第十名', '定位胆',
            '定位胆_第1~5名', '定位胆_第6~10名'
        ]
        
        number_group = group[group['玩法分类'].isin(number_categories)]
        
        for _, row in number_group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器解析位置和号码
            bets_by_position = ContentParser.parse_pk10_content(content)
            
            for position, numbers in bets_by_position.items():
                if numbers:  # 如果有号码投注
                    # 记录具体号码
                    for num in numbers:
                        position_bets[position].add(f'号码{num}')
    
    def _extract_position_from_independent_category(self, category):
        """从独立玩法分类中提取位置"""
        category_str = str(category)
        
        position_mapping = {
            '冠军': ['冠军', '前一'],
            '亚军': ['亚军'],
            '季军': ['季军', '第三名'],
            '第四名': ['第四名'],
            '第五名': ['第五名'],
            '第六名': ['第六名'],
            '第七名': ['第七名'],
            '第八名': ['第八名'],
            '第九名': ['第九名'],
            '第十名': ['第十名']
        }
        
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                if keyword in category_str:
                    return position
        
        return None
    
    def _extract_detailed_bets_from_content(self, content, position_bets):
        """从内容中提取详细的投注信息"""
        content_str = str(content)
        
        if '-' in content_str:
            parts = content_str.split(',')
            for part in parts:
                if '-' in part:
                    try:
                        position, bet_option = part.split('-', 1)
                        position = self.data_analyzer._normalize_pk10_position(position)
                        bet_option = bet_option.strip()
                        
                        # 记录具体的投注类型
                        if bet_option in ['大', '小', '单', '双']:
                            position_bets[position].add(bet_option)
                    except ValueError:
                        continue
    
    def _analyze_identical_bets_detailed(self, common_bets):
        """分析十个位置相同的投注内容 - 详细版本"""
        bet_list = list(common_bets)
        
        # 如果是单一投注类型
        if len(bet_list) == 1:
            bet = bet_list[0]
            if bet in ['大', '小', '单', '双']:
                return f"全部投注{bet}"
            elif bet.startswith('号码'):
                # 提取号码，如'号码5' -> '号码5'
                return f"全部投注{bet}"
            else:
                return f"全部投注{bet}"
        
        # 如果是组合投注（如同时投注大和单）
        else:
            # 先排序，确保显示一致
            sorted_bets = sorted(bet_list)
            # 分离号码投注和其他投注
            number_bets = [bet for bet in sorted_bets if bet.startswith('号码')]
            other_bets = [bet for bet in sorted_bets if not bet.startswith('号码')]
            
            descriptions = []
            if other_bets:
                descriptions.append("、".join(other_bets))
            if number_bets:
                # 如果有多个号码，显示所有号码
                number_desc = "、".join(number_bets)
                descriptions.append(number_desc)
            
            return "，".join(descriptions)
    
    def _collect_position_bets_from_plays(self, account, lottery, period, group, all_position_bets):
        """从各种玩法中收集位置投注信息 - 增强版本，记录具体投注内容"""
        
        # 1. 从两面玩法收集
        two_sides_categories = ['两面', '双面']
        two_sides_group = group[group['玩法分类'].isin(two_sides_categories)]
        
        for _, row in two_sides_group.iterrows():
            content = str(row['内容'])
            self._extract_position_bets_from_content(content, all_position_bets)
        
        # 2. 从独立玩法收集（大小单双龙虎）
        independent_categories = [
            '大小_冠军', '大小_亚军', '大小_季军',
            '单双_冠军', '单双_亚军', '单双_季军',
            '龙虎_冠军', '龙虎_亚军', '龙虎_季军'
        ]
        
        independent_group = group[group['玩法分类'].isin(independent_categories)]
        
        for _, row in independent_group.iterrows():
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
            
            # 提取具体的投注内容
            if '大小' in category:
                bets = self.data_analyzer.extract_size_parity_from_content(content)
                # 只关注大小
                size_bets = [bet for bet in bets if bet in ['大', '小']]
                if size_bets:
                    # 记录具体的投注内容，而不是笼统的"大小类"
                    all_position_bets[position].update(size_bets)
            elif '单双' in category:
                bets = self.data_analyzer.extract_size_parity_from_content(content)
                # 只关注单双
                parity_bets = [bet for bet in bets if bet in ['单', '双']]
                if parity_bets:
                    # 记录具体的投注内容，而不是笼统的"单双类"
                    all_position_bets[position].update(parity_bets)
        
        # 3. 从号码类玩法收集（定位胆等）
        number_categories = [
            '1-5名', '6-10名', '冠军', '前一', '亚军', '第三名', '第四名', '第五名',
            '第六名', '第七名', '第八名', '第九名', '第十名', '定位胆',
            '定位胆_第1~5名', '定位胆_第6~10名'
        ]
        
        number_group = group[group['玩法分类'].isin(number_categories)]
        
        for _, row in number_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 使用统一解析器解析位置
            bets_by_position = ContentParser.parse_pk10_content(content)
            
            for position, numbers in bets_by_position.items():
                if numbers:  # 如果有号码投注
                    all_position_bets[position].add('号码')
    
    def _extract_position_bets_from_content(self, content, all_position_bets):
        """从内容中提取位置投注信息 - 增强版本，记录具体投注内容"""
        content_str = str(content)
        
        if '-' in content_str:
            parts = content_str.split(',')
            for part in parts:
                if '-' in part:
                    try:
                        position, bet_option = part.split('-', 1)
                        position = self.data_analyzer._normalize_pk10_position(position)
                        bet_option = bet_option.strip()
                        
                        # 直接记录具体的投注类型，而不是分类
                        if bet_option in ['大', '小', '单', '双']:
                            all_position_bets[position].add(bet_option)
                    except ValueError:
                        continue
    
    def _analyze_bet_types(self, all_position_bets, standard_positions):
        """分析投注类型 - 最终修复版本"""
        # 统计每个具体投注类型的数量
        size_bets_count = {'大': 0, '小': 0}
        parity_bets_count = {'单': 0, '双': 0}
        number_count = 0
        
        for position in standard_positions:
            if position in all_position_bets:
                bets = all_position_bets[position]
                
                # 统计具体的大小投注
                if '大' in bets:
                    size_bets_count['大'] += 1
                if '小' in bets:
                    size_bets_count['小'] += 1
                
                # 统计具体的单双投注
                if '单' in bets:
                    parity_bets_count['单'] += 1
                if '双' in bets:
                    parity_bets_count['双'] += 1
                
                # 号码类投注（如果有的话）
                if '号码' in bets:
                    number_count += 1
        
        # 构建准确的投注类型描述
        bet_types = []
        
        # 大小投注：只有当一个类型在8个或以上位置出现时才显示
        for size_type, count in size_bets_count.items():
            if count >= 8:
                bet_types.append(size_type)
                break  # 只显示主要的大小类型
        
        # 单双投注：只有当一个类型在8个或以上位置出现时才显示
        for parity_type, count in parity_bets_count.items():
            if count >= 8:
                bet_types.append(parity_type)
                break  # 只显示主要的单双类型
        
        # 号码投注
        if number_count >= 8:
            bet_types.append('号码')
        
        return '、'.join(bet_types) if bet_types else '混合投注'

    # =============== 时时彩分析方法 ===============
    def analyze_ssc_patterns(self, df, period_amount_dict):
        """分析时时彩投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == 'SSC']
        
        if len(df_target) == 0:
            return results
        
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            self._analyze_ssc_two_sides(account, lottery, period, group, results, period_amount)
            self._analyze_ssc_douniu(account, lottery, period, group, results, period_amount)
            self._analyze_ssc_dingwei(account, lottery, period, group, results, period_amount)
            self._analyze_ssc_zonghe(account, lottery, period, group, results, period_amount)
            self._analyze_ssc_dingwei_detailed(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_ssc_two_sides(self, account, lottery, period, group, results, period_amount):
        two_sides_group = group[group['玩法分类'] == '两面']
        
        total_bets = set()
        ball_bets = defaultdict(set)
        
        for _, row in two_sides_group.iterrows():
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
                '玩法分类': '两面',
                '矛盾类型': '、'.join(conflicts),
                '投注内容': f"总和:{','.join(sorted(total_bets))}",
                '当期投注金额': period_amount,  # 添加金额信息
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
                    '玩法分类': '两面',
                    '矛盾类型': f"{ball}{'、'.join(ball_conflicts)}",
                    '投注内容': f"{ball}:{','.join(sorted(bets))}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': f"{ball}{'、'.join(ball_conflicts)}"}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
    
    def _analyze_ssc_douniu(self, account, lottery, period, group, results, period_amount):
        """分析时时彩斗牛玩法 - 修正版：只检测多码，不检测全包"""
        douniu_group = group[group['玩法分类'] == '斗牛']
        
        if douniu_group.empty:
            return
        
        # 使用集合收集所有斗牛类型
        all_bull_types = set()
        
        for _, row in douniu_group.iterrows():
            content = str(row['内容'])
            
            # 提取斗牛类型
            bull_types = self.data_analyzer.extract_douniu_types(content)
            if bull_types:
                all_bull_types.update(bull_types)
        
        # 斗牛多码检测（使用配置的阈值）
        if len(all_bull_types) >= THRESHOLD_CONFIG['SSC']['douniu_multi']:
            # 斗牛多码检测
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '斗牛',
                '违规类型': '斗牛多码',
                '号码数量': len(all_bull_types),
                '投注内容': f"斗牛多类型投注: {', '.join(sorted(all_bull_types))}",
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_bull_types)}, '斗牛多码')
            }
            self._add_unique_result(results, '斗牛多码', record)
    
    def _analyze_ssc_dingwei(self, account, lottery, period, group, results, period_amount):
        dingwei_categories = ['定位胆', '1-5球', '第1球', '第2球', '第3球', '第4球', '第5球']
        
        dingwei_group = group[group['玩法分类'].isin(dingwei_categories)]
        
        position_numbers = defaultdict(set)
        
        for _, row in dingwei_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 识别彩种类型
            lottery_type = self.identify_lottery_type(lottery)
            
            # PK10竖线分隔格式处理
            if lottery_type == 'PK10' and '|' in content and re.search(r'\d{2}', content):
                positions = ['冠军', '亚军', '第三名', '第四名', '第五名']
                parts = content.split('|')
                
                for i, part in enumerate(parts):
                    if i < len(positions):
                        position = positions[i]
                        numbers = self.data_analyzer.extract_numbers_from_content(part, 1, 10)
                        position_numbers[position].update(numbers)
            
            # 时时彩竖线分隔格式处理
            elif '|' in content:
                parts = content.split('|')
                positions = ['第1球', '第2球', '第3球', '第4球', '第5球']
                for i, part in enumerate(parts):
                    if i < len(positions) and part.strip() and part.strip() != '_':
                        numbers = self.data_analyzer.extract_numbers_from_content(part, 0, 9)
                        position_numbers[positions[i]].update(numbers)
            
            elif '-' in content:
                parts = content.split(',')
                for part in parts:
                    if '-' in part:
                        position, numbers_str = part.split('-', 1)
                        numbers = self.data_analyzer.extract_numbers_from_content(numbers_str, 0, 9)
                        position_numbers[position].update(numbers)
            
            else:
                numbers = self.data_analyzer.extract_numbers_from_content(content, 0, 9)
                if numbers:
                    position = '第1球'
                    position_numbers[position].update(numbers)
        
        for position, numbers in position_numbers.items():
            if len(numbers) >= THRESHOLD_CONFIG['SSC']['dingwei_multi']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '定位胆',
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([str(num) for num in sorted(numbers)])}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '定位胆多码')
                }
                self._add_unique_result(results, '定位胆多码', record)
    
    def _analyze_ssc_zonghe(self, account, lottery, period, group, results, period_amount):
        zonghe_group = group[group['玩法分类'] == '总和']
        
        all_bets = set()
        
        for _, row in zonghe_group.iterrows():
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
                '玩法分类': '总和',
                '矛盾类型': '、'.join(conflicts),
                '投注内容': ', '.join(sorted(all_bets)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '总和矛盾')
            }
            self._add_unique_result(results, '总和矛盾', record)
    
    def _analyze_ssc_dingwei_detailed(self, account, lottery, period, group, results, period_amount):
        """时时彩定位胆细分位置检测 - 增强位置判断"""
        dingwei_detailed_categories = [
            '定位_万位', '定位_千位', '定位_百位', '定位_十位', '定位_个位',
            '万位', '千位', '百位', '十位', '个位',
            '第1球', '第2球', '第3球', '第4球', '第5球'
        ]
        
        dingwei_detailed_group = group[group['玩法分类'].isin(dingwei_detailed_categories)]
        
        position_numbers = defaultdict(set)
        
        # 这个for循环需要缩进4个空格（在方法体内）
        for _, row in dingwei_detailed_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 新增：基于内容重新分类
            actual_category = self.normalize_play_category_from_content(content, category, 'SSC')
            
            # 增强位置判断：从玩法分类推断位置
            inferred_position = self._infer_ssc_position_from_category(actual_category)
            
            # 使用统一解析器
            bets_by_position = ContentParser.parse_ssc_content(content)
            
            for position, bets in bets_by_position.items():
                # 如果解析出的位置是"未知位置"，使用从玩法分类推断的位置
                if position == '未知位置' and inferred_position:
                    position = inferred_position
                
                # 提取每个位置的号码
                for bet in bets:
                    numbers = self.data_analyzer.extract_numbers_from_content(bet, 0, 9)
                    position_numbers[position].update(numbers)
        
        # 检查每个位置的超码
        for position, numbers in position_numbers.items():
            if len(numbers) >= THRESHOLD_CONFIG['SSC']['dingwei_multi']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': f'{position}多码',
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([str(num) for num in sorted(numbers)])}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '定位胆多码')
                }
                self._add_unique_result(results, '定位胆多码', record)
    
    def _infer_ssc_position_from_category(self, category):
        """从时时彩玩法分类推断位置"""
        category_str = str(category).strip()
        
        position_mapping = {
            '第1球': ['第1球', '定位_万位', '万位'],
            '第2球': ['第2球', '定位_千位', '千位'],
            '第3球': ['第3球', '定位_百位', '百位'],
            '第4球': ['第4球', '定位_十位', '十位'],
            '第5球': ['第5球', '定位_个位', '个位']
        }
        
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                if keyword in category_str:
                    return position
        
        return None

    # =============== 六合彩分析方法 ===============
    def analyze_lhc_patterns(self, df, period_amount_dict):
        """分析六合彩投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == 'LHC']
        
        if len(df_target) == 0:
            return results
        
        # 使用独立的尾数检测方法
        self._analyze_lhc_tail_plays(df_target, results, period_amount_dict)
        
        # 其他检测方法
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            # 先调用其他检测方法
            self._analyze_lhc_tema(account, lottery, period, group, results, period_amount)  # 特码多码检测
            self._analyze_lhc_two_sides(account, lottery, period, group, results, period_amount)  # 两面玩法矛盾、区间多组等
            
            # 然后再调用变相超码检测（它需要知道其他检测的结果）
            self._analyze_lhc_tema_contradiction(account, lottery, period, group, results, period_amount)
            
            # 其他检测方法
            self._analyze_lhc_lianxiao(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_lianwei(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_zhengma_wave_detailed(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_zhengma(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_zhengma_1_6(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_zhengte(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_pingte(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_texiao(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_yixiao(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_wave(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_five_elements(account, lottery, period, group, results, period_amount)
            self._analyze_lhc_banbo(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_lhc_tail_plays(self, df_target, results, period_amount_dict):
        """分析六合彩尾数玩法的完整逻辑 - 从Colab版本移植"""
        tail_categories = ['尾数', '尾数_头尾数', '特尾', '全尾']
        
        # 按不同尾数分类分别分析
        for tail_category in tail_categories:
            grouped = df_target[df_target['玩法分类'] == tail_category].groupby(
                ['会员账号', '彩种', '期号']
            )
            
            for (account, lottery, period), group in grouped:
                # 获取该期投注金额
                period_key = f"{account}_{lottery}_{period}"
                period_amount = period_amount_dict.get(period_key, 0.0)  # 从字典获取金额
                
                # 使用字典按调整后的分类聚合尾数
                category_tails = defaultdict(set)
                category_contents = defaultdict(list)
                
                for _, row in group.iterrows():
                    content = str(row['内容'])
                    category = str(row['玩法分类'])
                    
                    # 新增：基于内容重新分类
                    actual_category = self.normalize_play_category_from_content(content, category, 'LHC')
                    
                    clean_content = self.data_analyzer.parse_lhc_special_content(content)
                    tails = self.data_analyzer.extract_tails_from_content(clean_content)
                    category_tails[actual_category].update(tails)
                    category_contents[actual_category].append(clean_content)
                
                # 对每个调整后的分类分别检查阈值
                for actual_category, tails_set in category_tails.items():
                    if len(tails_set) >= THRESHOLD_CONFIG['LHC']['tail_play']:
                        # 根据不同的尾数分类，使用不同的结果键名
                        if actual_category == '尾数':
                            result_key = '尾数多码'
                        elif actual_category == '尾数_头尾数':
                            result_key = '尾数头尾多码'
                        elif actual_category == '特尾':
                            result_key = '特尾多尾'
                        elif actual_category == '全尾':
                            result_key = '全尾多尾'
                        else:
                            result_key = '尾数多码'
                        
                        # 构建投注内容显示 - 显示具体的尾数列表
                        bet_content = ', '.join([f"{tail}尾" for tail in sorted(tails_set)])
                        
                        record = {
                            '会员账号': account,
                            '彩种': lottery,
                            '期号': period,
                            '玩法分类': f"{actual_category}（{', '.join([str(tail) for tail in sorted(tails_set)])}）",
                            '尾数数量': len(tails_set),
                            '号码数量': len(tails_set),  # 兼容字段
                            '投注内容': bet_content,
                            '当期投注金额': period_amount,  # 使用从字典获取的金额
                            '排序权重': self._calculate_sort_weight({'尾数数量': len(tails_set)}, result_key)
                        }
                        self._add_unique_result(results, result_key, record)
    
    def _analyze_lhc_tema(self, account, lottery, period, group, results, period_amount):
        """分析六合彩特码玩法 - 添加金额参数"""
        tema_group = group[group['玩法分类'] == '特码']
        
        all_numbers = set()
        
        for _, row in tema_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            numbers = self.data_analyzer.extract_numbers_from_content(
                clean_content, 1, 49
            )
            all_numbers.update(numbers)
        
        if len(all_numbers) >= THRESHOLD_CONFIG['LHC']['number_play']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '特码',
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([f"{num:02d}" for num in sorted(all_numbers)]),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '特码多码')
            }
            self._add_unique_result(results, '特码多码', record)

    def _analyze_lhc_tema_contradiction(self, account, lottery, period, group, results, period_amount):
        """分析六合彩特码变相超码 - 增强版：添加色波和半波检测"""
        
        # ==================== 首先检查是否已经触发了其他违规检测 ====================
        
        # 检查是否已经检测到以下违规类型
        skip_contradiction = False
        skip_interval_contradiction = False
        
        # 检查是否已经检测到特码多码
        if '特码多码' in results and any(
            r['会员账号'] == account and r['彩种'] == lottery and r['期号'] == period 
            for r in results['特码多码']
        ):
            skip_contradiction = True
        
        # 检查是否已经检测到两面玩法矛盾
        if '两面玩法矛盾' in results and any(
            r['会员账号'] == account and r['彩种'] == lottery and r['期号'] == period 
            for r in results['两面玩法矛盾']
        ):
            skip_contradiction = True
        
        # 检查是否已经检测到区间多组
        if '区间多组' in results and any(
            r['会员账号'] == account and r['彩种'] == lottery and r['期号'] == period 
            for r in results['区间多组']
        ):
            skip_interval_contradiction = True
        
        # ==================== 情况1：检测特码玩法（号码）+ 两面玩法（大小单双）+ 波色/半波 ====================
        
        # 如果已经触发特码多码或两面玩法矛盾，则跳过第一种情况的检测
        if not skip_contradiction:
            # 从特码玩法中提取号码
            tema_group = group[group['玩法分类'] == '特码']
            
            if not tema_group.empty:
                # 收集所有特码投注的号码
                all_numbers = set()
                
                for _, row in tema_group.iterrows():
                    content = str(row['内容'])
                    
                    # 解析内容，提取号码
                    clean_content = self.data_analyzer.parse_lhc_special_content(content)
                    numbers = self.data_analyzer.extract_numbers_from_content(clean_content, 1, 49)
                    all_numbers.update(numbers)
                
                # 检查是否达到特码多码阈值（31码）- 如果达到，则跳过变相超码检测
                if len(all_numbers) >= THRESHOLD_CONFIG['LHC']['number_play']:
                    # 已经达到特码多码阈值，由 _analyze_lhc_tema 方法处理，这里跳过
                    pass
                else:
                    # 从两面玩法中提取特码相关的大小单双和波色投注
                    two_sides_group = group[group['玩法分类'] == '两面']
                    
                    # 从色波/半波玩法中提取波色投注
                    wave_play_group = group[group['玩法分类'].isin(['色波', '半波', '红波', '蓝波', '绿波'])]
                    
                    # 初始化检测标志
                    has_big = False
                    has_small = False
                    has_single = False
                    has_double = False
                    
                    # 波色检测标志
                    has_red_wave = False
                    has_blue_wave = False
                    has_green_wave = False
                    
                    # 半波检测标志
                    has_red_big = False
                    has_red_small = False
                    has_red_single = False
                    has_red_double = False
                    has_blue_big = False
                    has_blue_small = False
                    has_blue_single = False
                    has_blue_double = False
                    has_green_big = False
                    has_green_small = False
                    has_green_single = False
                    has_green_double = False
                    
                    # 检查两面玩法中的投注
                    for _, row in two_sides_group.iterrows():
                        content = str(row['内容'])
                        
                        # 使用extract_lhc_two_sides_content方法解析内容
                        two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
                        
                        # 检查普通大小（normal_size）和普通单双（parity）
                        normal_size = two_sides_analysis.get('normal_size', set())
                        parity = two_sides_analysis.get('parity', set())
                        
                        if '大' in normal_size:
                            has_big = True
                        if '小' in normal_size:
                            has_small = True
                        if '单' in parity:
                            has_single = True
                        if '双' in parity:
                            has_double = True
                    
                    # 检查波色和半波玩法中的投注
                    for _, row in wave_play_group.iterrows():
                        content = str(row['内容'])
                        content_lower = content.lower()
                        
                        # 解析玩法-投注内容格式
                        if '-' in content:
                            parts = content.split('-', 1)
                            bet_content = parts[1].strip()
                        else:
                            bet_content = content
                        
                        bet_content_lower = bet_content.lower()
                        
                        # 检测波色
                        if '红波' in bet_content_lower or '红' == bet_content_lower or '红色' in bet_content_lower:
                            has_red_wave = True
                        if '蓝波' in bet_content_lower or '蓝' == bet_content_lower or '蓝色' in bet_content_lower:
                            has_blue_wave = True
                        if '绿波' in bet_content_lower or '绿' == bet_content_lower or '绿色' in bet_content_lower:
                            has_green_wave = True
                        
                        # 检测半波
                        if '红大' in bet_content_lower:
                            has_red_big = True
                        if '红小' in bet_content_lower:
                            has_red_small = True
                        if '红单' in bet_content_lower:
                            has_red_single = True
                        if '红双' in bet_content_lower:
                            has_red_double = True
                        if '蓝大' in bet_content_lower:
                            has_blue_big = True
                        if '蓝小' in bet_content_lower:
                            has_blue_small = True
                        if '蓝单' in bet_content_lower:
                            has_blue_single = True
                        if '蓝双' in bet_content_lower:
                            has_blue_double = True
                        if '绿大' in bet_content_lower:
                            has_green_big = True
                        if '绿小' in bet_content_lower:
                            has_green_small = True
                        if '绿单' in bet_content_lower:
                            has_green_single = True
                        if '绿双' in bet_content_lower:
                            has_green_double = True
                    
                    # 检测第一种情况：有号码投注且有大小单双或波色/半波投注
                    if all_numbers and (has_big or has_small or has_single or has_double or 
                                        has_red_wave or has_blue_wave or has_green_wave or
                                        has_red_big or has_red_small or has_red_single or has_red_double or
                                        has_blue_big or has_blue_small or has_blue_single or has_blue_double or
                                        has_green_big or has_green_small or has_green_single or has_green_double):
                        
                        # 计算号码的属性分布
                        small_values = [num for num in all_numbers if 1 <= num <= 24]  # 六合彩小：1-24
                        big_values = [num for num in all_numbers if 25 <= num <= 49]   # 六合彩大：25-49
                        single_values = [num for num in all_numbers if num % 2 == 1]   # 单数
                        double_values = [num for num in all_numbers if num % 2 == 0]   # 双数
                        
                        # 定义波色号码范围
                        red_wave_numbers = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
                        blue_wave_numbers = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
                        green_wave_numbers = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}
                        
                        # 计算波色分布
                        red_values = [num for num in all_numbers if num in red_wave_numbers]
                        blue_values = [num for num in all_numbers if num in blue_wave_numbers]
                        green_values = [num for num in all_numbers if num in green_wave_numbers]
                        
                        # 计算半波分布
                        # 红大：红波且大
                        red_big_values = [num for num in red_values if num >= 25]
                        red_small_values = [num for num in red_values if num <= 24]
                        red_single_values = [num for num in red_values if num % 2 == 1]
                        red_double_values = [num for num in red_values if num % 2 == 0]
                        
                        # 蓝大：蓝波且大
                        blue_big_values = [num for num in blue_values if num >= 25]
                        blue_small_values = [num for num in blue_values if num <= 24]
                        blue_single_values = [num for num in blue_values if num % 2 == 1]
                        blue_double_values = [num for num in blue_values if num % 2 == 0]
                        
                        # 绿大：绿波且大
                        green_big_values = [num for num in green_values if num >= 25]
                        green_small_values = [num for num in green_values if num <= 24]
                        green_single_values = [num for num in green_values if num % 2 == 1]
                        green_double_values = [num for num in green_values if num % 2 == 0]
                        
                        # 收集所有可能的矛盾
                        possible_contradictions = []
                        
                        # 1. 投注小但包含多个大号码
                        if has_small and len(big_values) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(big_values)
                            description = f"投注小但包含多个大号码(小{len(small_values)}个,大{len(big_values)}个)"
                            possible_contradictions.append(('大小矛盾', description, contradiction_value))
                        
                        # 2. 投注大但包含多个小号码
                        if has_big and len(small_values) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(small_values)
                            description = f"投注大但包含多个小号码(小{len(small_values)}个,大{len(big_values)}个)"
                            possible_contradictions.append(('大小矛盾', description, contradiction_value))
                        
                        # 3. 投注单但包含多个双号码
                        if has_single and len(double_values) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(double_values)
                            description = f"投注单但包含多个双号码(单{len(single_values)}个,双{len(double_values)}个)"
                            possible_contradictions.append(('单双矛盾', description, contradiction_value))
                        
                        # 4. 投注双但包含多个单号码
                        if has_double and len(single_values) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(single_values)
                            description = f"投注双但包含多个单号码(单{len(single_values)}个,双{len(double_values)}个)"
                            possible_contradictions.append(('单双矛盾', description, contradiction_value))
                        
                        # 5. 投注红波但包含多个非红波号码（蓝波或绿波）
                        if has_red_wave and (len(blue_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(blue_values) + len(green_values)
                            description = f"投注红波但包含多个非红波号码(红{len(red_values)}个,蓝{len(blue_values)}个,绿{len(green_values)}个)"
                            possible_contradictions.append(('波色矛盾', description, contradiction_value))
                        
                        # 6. 投注蓝波但包含多个非蓝波号码（红波或绿波）
                        if has_blue_wave and (len(red_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_values) + len(green_values)
                            description = f"投注蓝波但包含多个非蓝波号码(红{len(red_values)}个,蓝{len(blue_values)}个,绿{len(green_values)}个)"
                            possible_contradictions.append(('波色矛盾', description, contradiction_value))
                        
                        # 7. 投注绿波但包含多个非绿波号码（红波或蓝波）
                        if has_green_wave and (len(red_values) + len(blue_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_values) + len(blue_values)
                            description = f"投注绿波但包含多个非绿波号码(红{len(red_values)}个,蓝{len(blue_values)}个,绿{len(green_values)}个)"
                            possible_contradictions.append(('波色矛盾', description, contradiction_value))
                        
                        # 8. 半波投注矛盾检测
                        # 红大半波
                        if has_red_big and (len(red_small_values) + len(blue_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_small_values) + len(blue_values) + len(green_values)
                            description = f"投注红大半波但包含多个非红大号码"
                            possible_contradictions.append(('半波矛盾', description, contradiction_value))
                        
                        # 红小半波
                        if has_red_small and (len(red_big_values) + len(blue_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_big_values) + len(blue_values) + len(green_values)
                            description = f"投注红小半波但包含多个非红小号码"
                            possible_contradictions.append(('半波矛盾', description, contradiction_value))
                        
                        # 红单单波
                        if has_red_single and (len(red_double_values) + len(blue_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_double_values) + len(blue_values) + len(green_values)
                            description = f"投注红单半波但包含多个非红单号码"
                            possible_contradictions.append(('半波矛盾', description, contradiction_value))
                        
                        # 红双半波
                        if has_red_double and (len(red_single_values) + len(blue_values) + len(green_values)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                            contradiction_value = len(red_single_values) + len(blue_values) + len(green_values)
                            description = f"投注红双半波但包含多个非红双号码"
                            possible_contradictions.append(('半波矛盾', description, contradiction_value))
                        
                        # 类似地添加其他半波的检测...
                        
                        # 如果有检测到矛盾，创建记录
                        if possible_contradictions:
                            # 按矛盾值降序排序
                            possible_contradictions.sort(key=lambda x: x[2], reverse=True)
                            
                            # 选择矛盾值最大的那个
                            best_contradiction = possible_contradictions[0]
                            contradiction_type, contradiction_desc, contradiction_value = best_contradiction
                            
                            # 构建投注内容显示
                            bet_content_parts = []
                            if has_big:
                                bet_content_parts.append('大')
                            if has_small:
                                bet_content_parts.append('小')
                            if has_single:
                                bet_content_parts.append('单')
                            if has_double:
                                bet_content_parts.append('双')
                            if has_red_wave:
                                bet_content_parts.append('红波')
                            if has_blue_wave:
                                bet_content_parts.append('蓝波')
                            if has_green_wave:
                                bet_content_parts.append('绿波')
                            # 添加半波投注
                            if has_red_big:
                                bet_content_parts.append('红大')
                            if has_red_small:
                                bet_content_parts.append('红小')
                            if has_red_single:
                                bet_content_parts.append('红单')
                            if has_red_double:
                                bet_content_parts.append('红双')
                            if has_blue_big:
                                bet_content_parts.append('蓝大')
                            if has_blue_small:
                                bet_content_parts.append('蓝小')
                            if has_blue_single:
                                bet_content_parts.append('蓝单')
                            if has_blue_double:
                                bet_content_parts.append('蓝双')
                            if has_green_big:
                                bet_content_parts.append('绿大')
                            if has_green_small:
                                bet_content_parts.append('绿小')
                            if has_green_single:
                                bet_content_parts.append('绿单')
                            if has_green_double:
                                bet_content_parts.append('绿双')
                            
                            bet_content = ', '.join(bet_content_parts)
                            
                            # 添加号码部分
                            if all_numbers:
                                numbers_content = ', '.join([f"{num:02d}" for num in sorted(all_numbers)])
                                bet_content += f" | 号码: {numbers_content}"
                            
                            record = {
                                '会员账号': account,
                                '彩种': lottery,
                                '期号': period,
                                '玩法分类': '特码',
                                '违规类型': '特码变相超码',
                                '矛盾类型': contradiction_desc,
                                '矛盾值': contradiction_value,
                                '投注内容': bet_content,
                                '号码数量': len(all_numbers),
                                '小号码数量': len(small_values),
                                '大号码数量': len(big_values),
                                '单号码数量': len(single_values),
                                '双号码数量': len(double_values),
                                '红波号码数量': len(red_values),
                                '蓝波号码数量': len(blue_values),
                                '绿波号码数量': len(green_values),
                                '当期投注金额': period_amount,
                                '排序权重': self._calculate_sort_weight(
                                    {'矛盾值': contradiction_value, '号码数量': len(all_numbers)}, 
                                    '特码变相超码'
                                )
                            }
                            self._add_unique_result(results, '特码变相超码', record)
        
        # ==================== 情况2：检测两面玩法中的区间投注+大小单双/波色/半波投注 ====================
        
        # 如果已经触发区间多组，则跳过第二种情况的检测
        if not skip_interval_contradiction:
            # 获取所有两面玩法的记录
            two_sides_group = group[group['玩法分类'] == '两面']
            
            if two_sides_group.empty:
                return
            
            # 检查是否有特码区间投注和特码大小/波色/半波投注
            has_interval_21_30 = False
            has_interval_31_40 = False
            has_interval_41_49 = False
            
            # 波色和半波投注标志
            has_red_wave = False
            has_blue_wave = False
            has_green_wave = False
            has_red_big = False
            has_red_small = False
            has_red_single = False
            has_red_double = False
            has_blue_big = False
            has_blue_small = False
            has_blue_single = False
            has_blue_double = False
            has_green_big = False
            has_green_small = False
            has_green_single = False
            has_green_double = False
            
            for _, row in two_sides_group.iterrows():
                content = str(row['内容'])
                
                # 检查特码区间投注
                if '特码两面' in content:
                    if '21-30' in content:
                        has_interval_21_30 = True
                    if '31-40' in content:
                        has_interval_31_40 = True
                    if '41-49' in content:
                        has_interval_41_49 = True
                
                # 检查特码波色和半波投注
                content_lower = content.lower()
                if '红波' in content_lower:
                    has_red_wave = True
                if '蓝波' in content_lower:
                    has_blue_wave = True
                if '绿波' in content_lower:
                    has_green_wave = True
                if '红大' in content_lower:
                    has_red_big = True
                if '红小' in content_lower:
                    has_red_small = True
                if '红单' in content_lower:
                    has_red_single = True
                if '红双' in content_lower:
                    has_red_double = True
                if '蓝大' in content_lower:
                    has_blue_big = True
                if '蓝小' in content_lower:
                    has_blue_small = True
                if '蓝单' in content_lower:
                    has_blue_single = True
                if '蓝双' in content_lower:
                    has_blue_double = True
                if '绿大' in content_lower:
                    has_green_big = True
                if '绿小' in content_lower:
                    has_green_small = True
                if '绿单' in content_lower:
                    has_green_single = True
                if '绿双' in content_lower:
                    has_green_double = True
            
            # 检查是否有区间投注和大小/波色/半波投注
            has_any_interval = has_interval_21_30 or has_interval_31_40 or has_interval_41_49
            has_any_side_bet = (has_red_wave or has_blue_wave or has_green_wave or
                               has_red_big or has_red_small or has_red_single or has_red_double or
                               has_blue_big or has_blue_small or has_blue_single or has_blue_double or
                               has_green_big or has_green_small or has_green_single or has_green_double)
            
            if has_any_interval and has_any_side_bet:
                # 构建区间包含的号码集合
                interval_numbers = set()
                
                if has_interval_21_30:
                    interval_numbers.update(range(21, 31))  # 21-30
                
                if has_interval_31_40:
                    interval_numbers.update(range(31, 41))  # 31-40
                
                if has_interval_41_49:
                    interval_numbers.update(range(41, 50))  # 41-49
                
                # 如果没有任何区间号码，直接返回
                if not interval_numbers:
                    return
                
                # 统计区间号码总数
                total_interval_numbers = len(interval_numbers)
                
                # 检查是否达到特码多码阈值（31码）- 如果达到，则跳过区间变相超码检测
                if total_interval_numbers >= THRESHOLD_CONFIG['LHC']['number_play']:
                    # 已经达到特码多码阈值，跳过区间变相超码检测
                    return
                
                # 定义波色号码范围
                red_wave_numbers = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
                blue_wave_numbers = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
                green_wave_numbers = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}
                
                # 计算区间号码的属性分布
                interval_small = [num for num in interval_numbers if 1 <= num <= 24]
                interval_big = [num for num in interval_numbers if 25 <= num <= 49]
                interval_single = [num for num in interval_numbers if num % 2 == 1]
                interval_double = [num for num in interval_numbers if num % 2 == 0]
                
                # 计算区间号码的波色分布
                interval_red = [num for num in interval_numbers if num in red_wave_numbers]
                interval_blue = [num for num in interval_numbers if num in blue_wave_numbers]
                interval_green = [num for num in interval_numbers if num in green_wave_numbers]
                
                # 收集可能的矛盾
                possible_contradictions = []
                
                # 波色矛盾检测
                if has_red_wave and (len(interval_blue) + len(interval_green)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                    contradiction_value = len(interval_blue) + len(interval_green)
                    description = f"投注红波但区间包含多个非红波号码(红{len(interval_red)}个,蓝{len(interval_blue)}个,绿{len(interval_green)}个)"
                    possible_contradictions.append(('波色矛盾', description, contradiction_value))
                
                if has_blue_wave and (len(interval_red) + len(interval_green)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                    contradiction_value = len(interval_red) + len(interval_green)
                    description = f"投注蓝波但区间包含多个非蓝波号码(红{len(interval_red)}个,蓝{len(interval_blue)}个,绿{len(interval_green)}个)"
                    possible_contradictions.append(('波色矛盾', description, contradiction_value))
                
                if has_green_wave and (len(interval_red) + len(interval_blue)) >= THRESHOLD_CONFIG['LHC']['value_size_contradiction']:
                    contradiction_value = len(interval_red) + len(interval_blue)
                    description = f"投注绿波但区间包含多个非绿波号码(红{len(interval_red)}个,蓝{len(interval_blue)}个,绿{len(interval_green)}个)"
                    possible_contradictions.append(('波色矛盾', description, contradiction_value))
                
                # 如果有检测到矛盾，创建记录
                if possible_contradictions:
                    # 按矛盾值降序排序
                    possible_contradictions.sort(key=lambda x: x[2], reverse=True)
                    
                    # 选择矛盾值最大的那个
                    best_contradiction = possible_contradictions[0]
                    contradiction_type, contradiction_desc, contradiction_value = best_contradiction
                    
                    # 构建投注内容显示
                    interval_desc_parts = []
                    if has_interval_21_30:
                        interval_desc_parts.append('21-30')
                    if has_interval_31_40:
                        interval_desc_parts.append('31-40')
                    if has_interval_41_49:
                        interval_desc_parts.append('41-49')
                    interval_desc = ', '.join(interval_desc_parts)
                    
                    bet_content_parts = []
                    if has_red_wave:
                        bet_content_parts.append('红波')
                    if has_blue_wave:
                        bet_content_parts.append('蓝波')
                    if has_green_wave:
                        bet_content_parts.append('绿波')
                    bet_content = f"区间: {interval_desc} | 波色: {', '.join(bet_content_parts)}"
                    
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': '两面',
                        '违规类型': '特码区间变相超码',
                        '矛盾类型': contradiction_desc,
                        '矛盾值': contradiction_value,
                        '投注内容': bet_content,
                        '号码数量': total_interval_numbers,
                        '区间号码': interval_desc,
                        '红波号码数量': len(interval_red),
                        '蓝波号码数量': len(interval_blue),
                        '绿波号码数量': len(interval_green),
                        '当期投注金额': period_amount,
                        '排序权重': self._calculate_sort_weight(
                            {'矛盾值': contradiction_value, '号码数量': total_interval_numbers}, 
                            '特码区间变相超码'
                        )
                    }
                    self._add_unique_result(results, '特码区间变相超码', record)
    
    def _analyze_lhc_two_sides(self, account, lottery, period, group, results, period_amount):
        two_sides_group = group[group['玩法分类'] == '两面']
        
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
        
        for _, row in two_sides_group.iterrows():
            content = str(row['内容'])
            
            two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
            
            for bet_type in two_sides_analysis:
                if bet_type in all_bets:
                    all_bets[bet_type].update(two_sides_analysis[bet_type])
        
        # 区间多组 - 修复：显示具体的区间内容
        if len(all_bets['range_bet']) >= THRESHOLD_CONFIG['LHC']['range_bet']:
            # 将区间集合转换为排序后的列表
            sorted_ranges = sorted(list(all_bets['range_bet']))
            bet_content = ', '.join(sorted_ranges)
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '两面',
                '投注区间数': len(all_bets['range_bet']),
                '投注区间': sorted_ranges,
                '投注内容': bet_content,  # 添加投注内容字段
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注区间数': len(all_bets['range_bet'])}, '区间多组')
            }
            self._add_unique_result(results, '区间多组', record)
        
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
                '玩法分类': '两面',
                '矛盾类型': '、'.join(conflict_types),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面玩法矛盾')
            }
            self._add_unique_result(results, '两面玩法矛盾', record)
        
        wave_set = all_bets.get('wave', set())
        if len(wave_set) >= THRESHOLD_CONFIG['LHC']['wave_bet']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '两面',
                '投注波色数': len(wave_set),
                '投注波色': sorted(list(wave_set)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注波色数': len(wave_set)}, '波色三组')
            }
            self._add_unique_result(results, '波色三组', record)
    
    def _analyze_lhc_zhengma(self, account, lottery, period, group, results, period_amount):
        zhengma_group = group[group['玩法分类'] == '正码']
        
        all_numbers = set()
        
        for _, row in zhengma_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            numbers = self.data_analyzer.extract_numbers_from_content(
                clean_content, 1, 49
            )
            all_numbers.update(numbers)
        
        if len(all_numbers) >= THRESHOLD_CONFIG['LHC']['number_play']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '正码',
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([f"{num:02d}" for num in sorted(all_numbers)]),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '正码多码')
            }
            self._add_unique_result(results, '正码多码', record)
    
    def _analyze_lhc_zhengma_1_6(self, account, lottery, period, group, results, period_amount):
        """六合彩正码1-6检测 - 增强位置判断"""
        zhengma_1_6_group = group[group['玩法分类'] == '正码1-6']
        
        if zhengma_1_6_group.empty:
            return
        
        position_bets = defaultdict(lambda: defaultdict(set))
        
        for _, row in zhengma_1_6_group.iterrows():
            content = str(row['内容'])
            
            # 使用统一解析器
            bets_by_position = ContentParser.parse_lhc_zhengma_content(content)
            
            for position, bets in bets_by_position.items():
                # 标准化位置名称
                normalized_position = self._normalize_zhengma_position(position)
                
                for bet in bets:
                    if bet == '合单':
                        position_bets[normalized_position]['sum_parity'].add('合单')
                    elif bet == '合双':
                        position_bets[normalized_position]['sum_parity'].add('合双')
                    # 可以添加其他投注类型的解析
            
            # 检查每个位置的矛盾
            for position, bets_by_type in position_bets.items():
                conflicts = []
                
                # 合数单双矛盾
                sum_parity_bets = bets_by_type.get('sum_parity', set())
                if '合单' in sum_parity_bets and '合双' in sum_parity_bets:
                    conflicts.append('合数单双矛盾')
                
                if conflicts:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': '正码1-6',
                        '位置': position,
                        '矛盾类型': '、'.join(conflicts),
                        '投注内容': f"{position}-{','.join(sorted(sum_parity_bets))}",
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '正码1-6矛盾')
                    }
                    self._add_unique_result(results, '正码1-6矛盾', record)
    
    def _normalize_zhengma_position(self, position):
        """标准化正码位置名称 - 修复版本"""
        position_mapping = {
            # 中文标准格式
            '正码一': '正码一', '正1': '正码一', '正码1': '正码一',
            '正码二': '正码二', '正2': '正码二', '正码2': '正码二', 
            '正码三': '正码三', '正3': '正码三', '正码3': '正码三',
            '正码四': '正码四', '正4': '正码四', '正码4': '正码四',
            '正码五': '正码五', '正5': '正码五', '正码5': '正码五',
            '正码六': '正码六', '正6': '正码六', '正码6': '正码六',
            # 处理可能的数字格式
            '1': '正码一', '2': '正码二', '3': '正码三',
            '4': '正码四', '5': '正码五', '6': '正码六',
            # 默认映射
            '未知位置': '正码一'
        }
        
        position = position.strip()
        
        # 直接映射
        if position in position_mapping:
            return position_mapping[position]
        
        # 模糊匹配
        for key, value in position_mapping.items():
            if key in position:
                return value
        
        # 如果包含数字，尝试提取数字并映射
        import re
        digit_match = re.search(r'\d', position)
        if digit_match:
            digit = digit_match.group()
            if digit in position_mapping:
                return position_mapping[digit]
        
        # 返回原位置，但确保至少是中文格式
        return position

    def _extract_specific_zhengte_position(self, content, category):
        """精确提取正特的具体位置"""
        content_str = str(content)
        category_str = str(category)
        
        # 位置映射
        position_mapping = {
            '正1特': ['正1特', '正一特', '正码一特', '正码1特'],
            '正2特': ['正2特', '正二特', '正码二特', '正码2特'],
            '正3特': ['正3特', '正三特', '正码三特', '正码3特'],
            '正4特': ['正4特', '正四特', '正码四特', '正码4特'],
            '正5特': ['正5特', '正五特', '正码五特', '正码5特'],
            '正6特': ['正6特', '正六特', '正码六特', '正码6特']
        }
        
        # 首先检查分类本身是否已经是具体位置
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                if keyword in category_str:
                    return position
        
        # 如果分类是"正特"，从内容中提取具体位置
        if category_str == '正特':
            for position, keywords in position_mapping.items():
                for keyword in keywords:
                    if keyword in content_str:
                        return position
            
            # 如果内容中包含数字，尝试推断位置
            if '正码一' in content_str or '正1' in content_str:
                return '正1特'
            elif '正码二' in content_str or '正2' in content_str:
                return '正2特'
            elif '正码三' in content_str or '正3' in content_str:
                return '正3特'
            elif '正码四' in content_str or '正4' in content_str:
                return '正4特'
            elif '正码五' in content_str or '正5' in content_str:
                return '正5特'
            elif '正码六' in content_str or '正6' in content_str:
                return '正6特'
        
        # 默认返回分类名称
        return category_str
    
    def _analyze_lhc_zhengte(self, account, lottery, period, group, results, period_amount):
        """分析六合彩正特玩法 - 改进版，精确识别具体位置"""
        zhengte_categories = ['正特', '正1特', '正2特', '正3特', '正4特', '正5特', '正6特']
        
        # 按具体位置分别统计
        position_numbers = defaultdict(set)
        position_bets = defaultdict(lambda: defaultdict(set))
        
        for category in zhengte_categories:
            category_group = group[group['玩法分类'] == category]
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                category = str(row['玩法分类'])
                
                # 精确识别具体位置
                specific_position = self._extract_specific_zhengte_position(content, category)
                
                clean_content = self.data_analyzer.parse_lhc_special_content(content)
                
                # 提取号码
                numbers = self.data_analyzer.extract_numbers_from_content(clean_content, 1, 49)
                position_numbers[specific_position].update(numbers)
                
                # 提取两面玩法内容
                two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
                for bet_type, bets in two_sides_analysis.items():
                    position_bets[specific_position][bet_type].update(bets)
        
        # 对每个具体位置分别进行检测
        for position, numbers in position_numbers.items():
            # 多号码检测
            if len(numbers) >= THRESHOLD_CONFIG['LHC']['number_play']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': f'{position}多码',
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}: {', '.join([f'{num:02d}' for num in sorted(numbers)])}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, f'{position}多码')
                }
                self._add_unique_result(results, f'{position}多码', record)
            
            # 矛盾投注检测
            bets_for_position = position_bets[position]
            conflicts = []
            
            if '大' in bets_for_position.get('normal_size', set()) and '小' in bets_for_position.get('normal_size', set()):
                conflicts.append('大小矛盾')
            if '单' in bets_for_position.get('parity', set()) and '双' in bets_for_position.get('parity', set()):
                conflicts.append('单双矛盾')
            if '尾大' in bets_for_position.get('tail_size', set()) and '尾小' in bets_for_position.get('tail_size', set()):
                conflicts.append('尾大小矛盾')
            if '合单' in bets_for_position.get('sum_parity', set()) and '合双' in bets_for_position.get('sum_parity', set()):
                conflicts.append('合数单双矛盾')
            
            if conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': position,
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, f'{position}矛盾')
                }
                self._add_unique_result(results, f'{position}矛盾', record)
    
    def _analyze_lhc_pingte(self, account, lottery, period, group, results, period_amount):
        pingte_group = group[group['玩法分类'] == '平特']
        
        all_zodiacs = set()
        
        for _, row in pingte_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            zodiacs = self.data_analyzer.extract_zodiacs_from_content(clean_content)
            all_zodiacs.update(zodiacs)
        
        if len(all_zodiacs) >= THRESHOLD_CONFIG['LHC']['zodiac_play']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '平特',
                '生肖数量': len(all_zodiacs),
                '投注内容': ', '.join(sorted(all_zodiacs)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'生肖数量': len(all_zodiacs)}, '平特多肖')
            }
            self._add_unique_result(results, '平特多肖', record)
    
    def _analyze_lhc_texiao(self, account, lottery, period, group, results, period_amount):
        texiao_group = group[group['玩法分类'] == '特肖']
        
        all_zodiacs = set()
        
        for _, row in texiao_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            zodiacs = self.data_analyzer.extract_zodiacs_from_content(clean_content)
            all_zodiacs.update(zodiacs)
        
        if len(all_zodiacs) >= THRESHOLD_CONFIG['LHC']['zodiac_play']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '特肖',
                '生肖数量': len(all_zodiacs),
                '投注内容': ', '.join(sorted(all_zodiacs)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'生肖数量': len(all_zodiacs)}, '特肖多肖')
            }
            self._add_unique_result(results, '特肖多肖', record)
    
    def _analyze_lhc_yixiao(self, account, lottery, period, group, results, period_amount):
        yixiao_group = group[group['玩法分类'] == '一肖']
        
        all_zodiacs = set()
        
        for _, row in yixiao_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            zodiacs = self.data_analyzer.extract_zodiacs_from_content(clean_content)
            all_zodiacs.update(zodiacs)
        
        if len(all_zodiacs) >= THRESHOLD_CONFIG['LHC']['zodiac_play']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '一肖',
                '生肖数量': len(all_zodiacs),
                '投注内容': ', '.join(sorted(all_zodiacs)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'生肖数量': len(all_zodiacs)}, '一肖多肖')
            }
            self._add_unique_result(results, '一肖多肖', record)
    
    def _analyze_lhc_wave(self, account, lottery, period, group, results, period_amount):
        """六合彩色波检测 - 包含半波内容检测，七色波就是色波"""
        wave_group = group[group['玩法分类'] == '色波']
        
        if wave_group.empty:
            return
        
        # 收集所有波色投注和半波投注
        all_wave_bets = set()
        all_banbo_bets = set()  # 半波投注
        
        # 定义半波投注项
        banbo_items = {
            '红大', '红小', '红单', '红双',
            '蓝大', '蓝小', '蓝单', '蓝双', 
            '绿大', '绿小', '绿单', '绿双'
        }
        
        for _, row in wave_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            
            # 提取传统波色
            waves = self.data_analyzer.extract_wave_color_from_content(clean_content)
            all_wave_bets.update(waves)
            
            # 提取半波投注项
            for item in banbo_items:
                if item in clean_content:
                    all_banbo_bets.add(item)
        
        # 检测1: 传统色波全包（红波、蓝波、绿波）- 七色波就是色波
        traditional_waves = {'红波', '蓝波', '绿波'}
        if traditional_waves.issubset(all_wave_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '色波',
                '违规类型': '色波全包',
                '投注波色数': len(traditional_waves),
                '投注波色': sorted(list(traditional_waves)),
                '投注内容': f"色波全包: {', '.join(sorted(traditional_waves))}",
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注波色数': len(traditional_waves)}, '色波全包')
            }
            self._add_unique_result(results, '色波全包', record)
        
        # 检测2: 色波玩法中的半波全包检测
        # 大小全包检测
        size_full_set = {'红大', '红小', '蓝大', '蓝小', '绿大', '绿小'}
        if size_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '色波',
                '违规类型': '色波中半波大小全包',
                '投注半波数': len(size_full_set),
                '投注半波': sorted(list(size_full_set)),
                '投注内容': ', '.join(sorted(size_full_set)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注半波数': len(size_full_set)}, '色波中半波大小全包')
            }
            self._add_unique_result(results, '色波中半波全包', record)
        
        # 单双全包检测
        parity_full_set = {'红单', '红双', '蓝单', '蓝双', '绿单', '绿双'}
        if parity_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '色波',
                '违规类型': '色波中半波单双全包',
                '投注半波数': len(parity_full_set),
                '投注半波': sorted(list(parity_full_set)),
                '投注内容': ', '.join(sorted(parity_full_set)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注半波数': len(parity_full_set)}, '色波中半波单双全包')
            }
            self._add_unique_result(results, '色波中半波全包', record)
    
    def _analyze_lhc_five_elements(self, account, lottery, period, group, results, period_amount):
        five_elements_group = group[group['玩法分类'] == '五行']
        
        all_elements = set()
        
        for _, row in five_elements_group.iterrows():
            content = str(row['内容'])
            clean_content = self.data_analyzer.parse_lhc_special_content(content)
            elements = self.data_analyzer.extract_five_elements_from_content(clean_content)
            all_elements.update(elements)
        
        if len(all_elements) >= THRESHOLD_CONFIG['LHC']['five_elements']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '五行',
                '投注五行数': len(all_elements),
                '投注五行': sorted(list(all_elements)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注五行数': len(all_elements)}, '五行多组')
            }
            self._add_unique_result(results, '五行多组', record)
    
    def _analyze_lhc_lianxiao(self, account, lottery, period, group, results, period_amount):
        """分析六合彩连肖玩法 - 修复版本，确保区分具体类型"""
        # 定义连肖类型及其对应的阈值
        lianxiao_config = {
            '二连肖': {'threshold': 7},
            '三连肖': {'threshold': 7},  
            '四连肖': {'threshold': 7},
            '五连肖': {'threshold': 8},
        }
        
        # 首先检查具体的连肖类型
        for lianxiao_type, config in lianxiao_config.items():
            lianxiao_group = group[group['玩法分类'] == lianxiao_type]
            
            for _, row in lianxiao_group.iterrows():
                content = str(row['内容'])
                category = str(row['玩法分类'])
                
                # 解析玩法-投注内容格式
                if '-' in content:
                    parts = content.split('-', 1)
                    bet_content = parts[1].strip()
                else:
                    bet_content = content
                    
                zodiacs = self.data_analyzer.extract_zodiacs_from_content(bet_content)
                
                # 使用针对具体连肖类型的阈值
                if len(zodiacs) >= config['threshold']:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': f"{lianxiao_type}（{len(zodiacs)}生肖）",
                        '违规类型': f'{lianxiao_type}多肖',
                        '生肖数量': len(zodiacs),
                        '投注内容': ', '.join(sorted(zodiacs)),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'生肖数量': len(zodiacs)}, f'{lianxiao_type}多肖')
                    }
                    self._add_unique_result(results, f'{lianxiao_type}多肖', record)
        
        # 然后检查通用的连肖类型（作为后备）
        generic_lianxiao_group = group[group['玩法分类'] == '连肖']
        if not generic_lianxiao_group.empty:
            # 尝试从内容中推断具体类型
            for _, row in generic_lianxiao_group.iterrows():
                content = str(row['内容'])
                
                # 从内容中推断具体连肖类型
                inferred_type = self._infer_lianxiao_type_from_content(content)
                
                # 解析玩法-投注内容格式
                if '-' in content:
                    parts = content.split('-', 1)
                    bet_content = parts[1].strip()
                else:
                    bet_content = content
                    
                zodiacs = self.data_analyzer.extract_zodiacs_from_content(bet_content)
                
                # 根据推断的类型使用相应的阈值，如果没有推断出类型则使用通用阈值
                if inferred_type and inferred_type in lianxiao_config:
                    threshold = lianxiao_config[inferred_type]['threshold']
                    display_type = inferred_type
                else:
                    threshold = 6  # 通用阈值
                    display_type = '连肖'
                
                if len(zodiacs) >= threshold:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': f"{display_type}（{len(zodiacs)}生肖）",
                        '违规类型': f'{display_type}多肖',
                        '生肖数量': len(zodiacs),
                        '投注内容': ', '.join(sorted(zodiacs)),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'生肖数量': len(zodiacs)}, f'{display_type}多肖')
                    }
                    self._add_unique_result(results, f'{display_type}多肖', record)
    
    def _infer_lianxiao_type_from_content(self, content):
        """从内容中推断连肖类型"""
        content_str = str(content)
        
        # 从内容中查找具体类型
        if '二连肖' in content_str:
            return '二连肖'
        elif '三连肖' in content_str:
            return '三连肖'
        elif '四连肖' in content_str:
            return '四连肖'
        elif '五连肖' in content_str:
            return '五连肖'
        
        return None
    
    def _analyze_lhc_lianwei(self, account, lottery, period, group, results, period_amount):
        """分析六合彩连尾玩法 - 修复版本，确保区分具体类型"""
        # 定义连尾类型及其对应的阈值
        lianwei_config = {
            '二连尾': {'threshold': 7},
            '三连尾': {'threshold': 7},
            '四连尾': {'threshold': 7},  
            '五连尾': {'threshold': 8},
        }
        
        # 首先检查具体的连尾类型
        for lianwei_type, config in lianwei_config.items():
            lianwei_group = group[group['玩法分类'] == lianwei_type]
            
            for _, row in lianwei_group.iterrows():
                content = str(row['内容'])
                tails = self.data_analyzer.extract_tails_from_content(content)
                
                # 使用针对具体连尾类型的阈值
                if len(tails) >= config['threshold']:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': f"{lianwei_type}（{len(tails)}尾）",
                        '违规类型': f'{lianwei_type}多尾',
                        '尾数数量': len(tails),
                        '投注内容': ', '.join([f"{tail}尾" for tail in sorted(tails)]),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'尾数数量': len(tails)}, f'{lianwei_type}多尾')
                    }
                    self._add_unique_result(results, f'{lianwei_type}多尾', record)
        
        # 然后检查通用的连尾类型（作为后备）
        generic_lianwei_group = group[group['玩法分类'] == '连尾']
        if not generic_lianwei_group.empty:
            # 尝试从内容中推断具体类型
            for _, row in generic_lianwei_group.iterrows():
                content = str(row['内容'])
                
                # 从内容中推断具体连尾类型
                inferred_type = self._infer_lianwei_type_from_content(content)
                
                tails = self.data_analyzer.extract_tails_from_content(content)
                
                # 根据推断的类型使用相应的阈值，如果没有推断出类型则使用通用阈值
                if inferred_type and inferred_type in lianwei_config:
                    threshold = lianwei_config[inferred_type]['threshold']
                    display_type = inferred_type
                else:
                    threshold = 6  # 通用阈值
                    display_type = '连尾'
                
                if len(tails) >= threshold:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': f"{display_type}（{len(tails)}尾）",
                        '违规类型': f'{display_type}多尾',
                        '尾数数量': len(tails),
                        '投注内容': ', '.join([f"{tail}尾" for tail in sorted(tails)]),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'尾数数量': len(tails)}, f'{display_type}多尾')
                    }
                    self._add_unique_result(results, f'{display_type}多尾', record)
    
    def _infer_lianwei_type_from_content(self, content):
        """从内容中推断连尾类型"""
        content_str = str(content)
        
        # 从内容中查找具体类型
        if '二连尾' in content_str:
            return '二连尾'
        elif '三连尾' in content_str:
            return '三连尾'
        elif '四连尾' in content_str:
            return '四连尾'
        elif '五连尾' in content_str:
            return '五连尾'
        
        return None
    
    def _analyze_lhc_zhengte_detailed(self, account, lottery, period, group, results):
        """六合彩正码特详细检测"""
        zhengte_categories = ['正1特', '正2特', '正3特', '正4特', '正5特', '正6特']
        
        for category in zhengte_categories:
            category_group = group[group['玩法分类'] == category]
            
            all_numbers = set()
            all_bets = defaultdict(set)
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                category = str(row['玩法分类'])
                
                # 新增：基于内容重新分类
                actual_category = self.normalize_play_category_from_content(content, category, 'LHC')
                
                clean_content = self.data_analyzer.parse_lhc_special_content(content)
                
                # 提取数字
                numbers = self.data_analyzer.extract_numbers_from_content(clean_content, 1, 49)
                all_numbers.update(numbers)
                
                # 提取两面玩法内容
                two_sides_analysis = self.data_analyzer.extract_lhc_two_sides_content(content)
                for bet_type, bets in two_sides_analysis.items():
                    all_bets[bet_type].update(bets)
            
            # 多号码检测
            if len(all_numbers) >= THRESHOLD_CONFIG['LHC']['number_play']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': category,
                    '号码数量': len(all_numbers),
                    '投注内容': ', '.join([f"{num:02d}" for num in sorted(all_numbers)]),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '正特多码')
                }
                self._add_unique_result(results, '正特多码', record)
            
            # 矛盾投注检测
            conflicts = []
            if '大' in all_bets.get('normal_size', set()) and '小' in all_bets.get('normal_size', set()):
                conflicts.append('大小矛盾')
            if '单' in all_bets.get('parity', set()) and '双' in all_bets.get('parity', set()):
                conflicts.append('单双矛盾')
            if '尾大' in all_bets.get('tail_size', set()) and '尾小' in all_bets.get('tail_size', set()):
                conflicts.append('尾大小矛盾')
            if '合单' in all_bets.get('sum_parity', set()) and '合双' in all_bets.get('sum_parity', set()):
                conflicts.append('合数单双矛盾')
            
            if conflicts:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': category,
                    '矛盾类型': '、'.join(conflict_types),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '正特矛盾')
                }
                self._add_unique_result(results, '正特矛盾', record)
    
    def _analyze_lhc_lianxiao_lianwei_detailed(self, account, lottery, period, group, results):
        """连肖连尾细分检测"""
        # 连肖细分
        lianxiao_categories = {
            '连肖连尾_二连肖': 2,
            '连肖连尾_三连肖': 3, 
            '连肖连尾_四连肖': 4,
            '连肖连尾_五连肖': 5
        }
        
        for category, threshold in lianxiao_categories.items():
            category_group = group[group['玩法分类'] == category]
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                category = str(row['玩法分类'])
                
                # 新增：基于内容重新分类
                actual_category = self.normalize_play_category_from_content(content, category, 'LHC')
                
                zodiacs = self.data_analyzer.extract_zodiacs_from_content(content)
                
                # 超过阈值检测
                if len(zodiacs) > threshold + 2:  # 允许一定的冗余
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': category,
                        '生肖数量': len(zodiacs),
                        '投注内容': ', '.join(sorted(zodiacs)),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'生肖数量': len(zodiacs)}, '连肖多肖')
                    }
                    self._add_unique_result(results, '连肖多肖', record)
        
        # 连尾细分
        lianwei_categories = {
            '连肖连尾_二连尾': 2,
            '连肖连尾_三连尾': 3,
            '连肖连尾_四连尾': 4,
            '连肖连尾_五连尾': 5
        }
        
        for category, threshold in lianwei_categories.items():
            category_group = group[group['玩法分类'] == category]
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                tails = self.data_analyzer.extract_tails_from_content(content)
                
                if len(tails) > threshold + 2:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': category,
                        '尾数数量': len(tails),
                        '投注内容': ', '.join([f"{tail}尾" for tail in sorted(tails)]),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'尾数数量': len(tails)}, '连尾多尾')
                    }
                    self._add_unique_result(results, '连尾多尾', record)
    
    def _analyze_lhc_banbo(self, account, lottery, period, group, results, period_amount):
        """六合彩半波检测 - 检测大小全包和单双全包，包括蓝波、绿波、红波玩法"""
        # 扩展半波相关的玩法分类
        banbo_categories = ['半波', '蓝波', '绿波', '红波']
        
        banbo_group = group[group['玩法分类'].isin(banbo_categories)]
        
        if banbo_group.empty:
            return
        
        # 定义两组半波全包
        size_full_set = {'红大', '红小', '蓝大', '蓝小', '绿大', '绿小'}  # 大小全包
        parity_full_set = {'红单', '红双', '蓝单', '蓝双', '绿单', '绿双'}  # 单双全包
        
        all_banbo_bets = set()
        
        for _, row in banbo_group.iterrows():
            content = str(row['内容'])
            
            # 解析玩法-投注内容格式
            if '-' in content:
                parts = content.split('-', 1)
                bet_content = parts[1].strip()  # 只使用投注内容部分
            else:
                bet_content = content
            
            # 提取所有半波投注项
            for bet in size_full_set.union(parity_full_set):
                if bet in bet_content:
                    all_banbo_bets.add(bet)
        
        # 检测大小全包
        if size_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '半波',
                '投注半波数': len(size_full_set),
                '投注半波': sorted(list(size_full_set)),
                '投注内容': ', '.join(sorted(size_full_set)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注半波数': len(size_full_set)}, '半波大小全包')
            }
            self._add_unique_result(results, '半波大小全包', record)
        
        # 检测单双全包
        if parity_full_set.issubset(all_banbo_bets):
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '半波',
                '投注半波数': len(parity_full_set),
                '投注半波': sorted(list(parity_full_set)),
                '投注内容': ', '.join(sorted(parity_full_set)),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注半波数': len(parity_full_set)}, '半波单双全包')
            }
            self._add_unique_result(results, '半波单双全包', record)

    def _analyze_lhc_zhengma_wave_detailed(self, account, lottery, period, group, results, period_amount):
        """分析六合彩正码中的波色投注 - 修复版本"""
        zhengma_categories = ['正码', '正码1-6', '正码一', '正码二', '正码三', '正码四', '正码五', '正码六']
        
        zhengma_group = group[group['玩法分类'].isin(zhengma_categories)]
        
        if zhengma_group.empty:
            return
        
        # 收集每个位置的波色投注 - 修复：确保位置准确
        position_waves = defaultdict(set)
        
        for _, row in zhengma_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 修复：优先从extracted_position字段获取位置（如果存在）
            if 'extracted_position' in row and pd.notna(row['extracted_position']):
                position = row['extracted_position']
            else:
                # 如果extracted_position不存在，则从玩法分类中提取
                position = self._extract_exact_position_from_category(category)
                if position == '未知位置':
                    # 最后从内容中提取
                    play_method, position, clean_content = self.enhanced_parser.extract_play_method_and_position(content, 'LHC')
            
            # 标准化位置名称
            normalized_position = self._normalize_zhengma_position_exact(position)
            
            # 从内容中提取波色
            waves = self._extract_wave_from_zhengma_content(content)
            if waves:
                position_waves[normalized_position].update(waves)
        
        # 检查每个位置的波色全包情况
        traditional_waves = {'红波', '蓝波', '绿波'}
        for position, waves in position_waves.items():
            # 如果该位置同时投注了红波、蓝波、绿波，则视为该位置波色全包
            if traditional_waves.issubset(waves):
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': f'{position}波色全包',
                    '位置': position,
                    '违规类型': f'{position}波色全包',
                    '投注波色数': len(traditional_waves),
                    '投注波色': sorted(list(traditional_waves)),
                    '投注内容': f"{position}波色全包: {', '.join(sorted(traditional_waves))}",
                    '排序权重': self._calculate_sort_weight({'投注波色数': len(traditional_waves)}, f'{position}波色全包')
                }
                self._add_unique_result(results, f'{position}波色全包', record)
        
    
    def _extract_wave_from_zhengma_bet(self, bet_content):
        """从正码投注内容中精确提取波色"""
        bet_str = str(bet_content).strip()
        waves = set()
        
        # 精确匹配波色关键词
        wave_keywords = {
            '红波': ['红波', '紅色波', '红'],
            '蓝波': ['蓝波', '藍波', '蓝', '藍'], 
            '绿波': ['绿波', '綠波', '绿', '綠']
        }
        
        for wave_name, keywords in wave_keywords.items():
            for keyword in keywords:
                # 精确匹配，避免部分匹配
                if keyword == bet_str or f"-{keyword}" in bet_str or f"{keyword}," in bet_str:
                    waves.add(wave_name)
                    break
        
        return waves

    # =============== 3D系列分析方法 ===============
    def analyze_3d_patterns(self, df, period_amount_dict):
        """分析3D系列投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == '3D']
        
        if len(df_target) == 0:
            return results
        
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            self._analyze_3d_two_sides(account, lottery, period, group, results, period_amount)
            self._analyze_3d_dingwei(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_3d_two_sides(self, account, lottery, period, group, results, period_amount):
        """分析3D两面玩法矛盾 - 增强竖线格式支持"""
        two_sides_group = group[group['玩法分类'] == '两面']
        
        if two_sides_group.empty:
            return
        
        # 按位置分类收集投注
        position_bets = defaultdict(set)
        
        for _, row in two_sides_group.iterrows():
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
        
        # 检查每个位置的矛盾（保持原有逻辑不变）
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
                    '玩法分类': '两面',
                    '位置': position,
                    '矛盾类型': '、'.join(conflicts),
                    '投注内容': f"{position}:{','.join(sorted(bet_options))}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflicts)}, '两面矛盾')
                }
                self._add_unique_result(results, '两面矛盾', record)
    
    def _analyze_3d_dingwei(self, account, lottery, period, group, results, period_amount):
        """分析3D定位胆多码 - 增强竖线格式支持"""
        dingwei_categories = ['定位胆', '定位胆_百位', '定位胆_十位', '定位胆_个位']
        
        dingwei_group = group[group['玩法分类'].isin(dingwei_categories)]
        
        position_numbers = defaultdict(set)
        
        # 修复这里的缩进：整个for循环应该缩进
        for _, row in dingwei_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            # 首先使用统一解析器解析竖线格式
            bets_by_position = self.data_analyzer.parse_3d_content(content)
            if bets_by_position:
                # 如果有解析结果，使用解析出的位置和号码
                for position, numbers in bets_by_position.items():
                    position_numbers[position].update(numbers)
                continue
            
            # 新增：基于内容重新分类（在原有逻辑之前）
            actual_category = self.normalize_play_category_from_content(content, category, '3D')
            
            # 如果没有竖线格式，使用原有逻辑
            # 确定位置
            if '百位' in actual_category:  # 这里要用 actual_category，不是 category
                position = '百位'
            elif '十位' in actual_category:  # 这里也要用 actual_category
                position = '十位'
            elif '个位' in actual_category:  # 这里也要用 actual_category
                position = '个位'
            else:
                # 从内容推断位置
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
            if len(numbers) >= THRESHOLD_CONFIG['3D']['dingwei_multi']:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': f'{position}多码',
                    '位置': position,
                    '号码数量': len(numbers),
                    '投注内容': f"{position}-{','.join([str(num) for num in sorted(numbers)])}",
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '定位胆多码')
                }
                self._add_unique_result(results, '定位胆多码', record)

    # =============== 快三分析方法 ===============
    def analyze_k3_patterns(self, df, period_amount_dict):
        """分析快三投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == 'K3']
        
        if len(df_target) == 0:
            return results
        
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            self._analyze_k3_hezhi_enhanced(account, lottery, period, group, results, period_amount)
            # 先进行聚合检测（更严格的检测）
            self._analyze_k3_dudan_aggregated(account, lottery, period, group, results, period_amount)
            # 如果聚合检测没有发现问题，再进行单个记录检测
            if not any('独胆多码' in key for key in results.keys()):
                self._analyze_k3_dudan(account, lottery, period, group, results, period_amount)
            self._analyze_k3_different(account, lottery, period, group, results, period_amount)
            self._analyze_k3_two_sides_plays(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_k3_hezhi_enhanced(self, account, lottery, period, group, results, period_amount):
        """分析快三和值玩法 - 优化版，避免重复检测"""
        hezhi_categories = ['和值', '和值_大小单双']
        
        hezhi_group = group[group['玩法分类'].isin(hezhi_categories)]
        
        if hezhi_group.empty:
            return
        
        all_numbers = set()
        all_contents = []
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in hezhi_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
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
        
        # 和值多码检测（11码或以上）- 如果检测到就完全退出
        if len(all_numbers) >= THRESHOLD_CONFIG['K3']['hezhi_multi_number']:
            bet_content = ', '.join([str(num) for num in sorted(all_numbers)])
            
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '和值',
                '号码数量': len(all_numbers),
                '投注内容': bet_content,
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '和值多码')
            }
            self._add_unique_result(results, '和值多码', record)
            return  # 完全退出，不进行后续检测
        
        # 和值矛盾检测（大小单双同时下注）
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
                '玩法分类': '和值',
                '矛盾类型': '、'.join(conflict_types),
                '投注内容': bet_content,
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '和值矛盾')
            }
            self._add_unique_result(results, '和值矛盾', record)
            return  # 如果检测到和值矛盾，也不进行和值变相超码检测
        
        # 和值变相超码检测 - 只有在没有检测到和值多码和和值矛盾时才进行
        if all_numbers and len(all_numbers) < THRESHOLD_CONFIG['K3']['hezhi_multi_number']:
            small_values = [num for num in all_numbers if 3 <= num <= 10]
            big_values = [num for num in all_numbers if 11 <= num <= 18]
            single_values = [num for num in all_numbers if num % 2 == 1]
            double_values = [num for num in all_numbers if num % 2 == 0]
            
            # 收集所有可能的矛盾
            possible_contradictions = []
            
            # 投注小但包含多个大号码（4个或以上）
            if has_small and len(big_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(big_values)
                description = f"投注小但包含多个大号码(小{len(small_values)}个,大{len(big_values)}个)"
                possible_contradictions.append(('大小矛盾', description, contradiction_value))
            
            # 投注大但包含多个小号码（4个或以上）
            if has_big and len(small_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(small_values)
                description = f"投注大但包含多个小号码(小{len(small_values)}个,大{len(big_values)}个)"
                possible_contradictions.append(('大小矛盾', description, contradiction_value))
            
            # 投注单但包含多个双号码（4个或以上）
            if has_single and len(double_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(double_values)
                description = f"投注单但包含多个双号码(单{len(single_values)}个,双{len(double_values)}个)"
                possible_contradictions.append(('单双矛盾', description, contradiction_value))
            
            # 投注双但包含多个单号码（4个或以上）
            if has_double and len(single_values) >= THRESHOLD_CONFIG['K3']['value_size_contradiction']:
                contradiction_value = len(single_values)
                description = f"投注双但包含多个单号码(单{len(single_values)}个,双{len(double_values)}个)"
                possible_contradictions.append(('单双矛盾', description, contradiction_value))
            
            # 优先展示数量最多的矛盾组合
            if possible_contradictions:
                # 按矛盾值降序排序
                possible_contradictions.sort(key=lambda x: x[2], reverse=True)
                
                # 选择矛盾值最大的那个
                best_contradiction = possible_contradictions[0]
                contradiction_type, contradiction_desc, contradiction_value = best_contradiction
                
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '和值',
                    '矛盾类型': contradiction_desc,
                    '矛盾值': contradiction_value,
                    '大号码数量': len(big_values),
                    '小号码数量': len(small_values),
                    '单号码数量': len(single_values),
                    '双号码数量': len(double_values),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'矛盾值': contradiction_value}, '和值变相超码')
                }
                self._add_unique_result(results, '和值变相超码', record)

    def _analyze_k3_dudan(self, account, lottery, period, group, results, period_amount):
        """分析快三独胆玩法 - 单个记录检测"""
        dudan_group = group[group['玩法分类'] == '独胆']
        
        for _, row in dudan_group.iterrows():
            content = str(row['内容'])
            category = str(row['玩法分类'])
            
            numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 6)
            
            # 检测单个记录的多号码（通常不会触发，因为三军是分开投注的）
            if len(numbers) >= 5:
                record = {
                    '会员账号': account,
                    '彩种': lottery,
                    '期号': period,
                    '玩法分类': '独胆',
                    '号码数量': len(numbers),
                    '投注内容': ', '.join([str(num) for num in sorted(numbers)]),
                    '当期投注金额': period_amount,  # 添加金额信息
                    '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '独胆多码')
                }
                self._add_unique_result(results, '独胆多码', record)
    
    def _analyze_k3_dudan_aggregated(self, account, lottery, period, group, results, period_amount):
        """分析快三独胆玩法 - 按账户期号聚合检测"""
        dudan_group = group[group['玩法分类'] == '独胆']
        
        if dudan_group.empty:
            return
        
        # 聚合同一账户同一期号的所有独胆投注
        all_numbers = set()
        
        for _, row in dudan_group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 6)
            all_numbers.update(numbers)
        
        # 使用配置的阈值
        threshold = THRESHOLD_CONFIG['K3'].get('dudan_multi_number', 3)
        if len(all_numbers) >= threshold:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '独胆',
                '号码数量': len(all_numbers),
                '投注内容': f"聚合投注: {', '.join([str(num) for num in sorted(all_numbers)])}",
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '独胆多码')
            }
            self._add_unique_result(results, '独胆多码', record)
    
    def _analyze_k3_different(self, account, lottery, period, group, results, period_amount):
        different_categories = ['二不同号', '三不同号']
        
        for category in different_categories:
            category_group = group[group['玩法分类'] == category]
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                numbers = self.data_analyzer.extract_numbers_from_content(content, 1, 6)
                
                if len(numbers) == 6:
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': category,
                        '号码数量': len(numbers),
                        '投注内容': ', '.join([str(num) for num in sorted(numbers)]),
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({'号码数量': len(numbers)}, '不同号全包')
                    }
                    self._add_unique_result(results, '不同号全包', record)
    
    def _analyze_k3_two_sides_plays(self, account, lottery, period, group, results, period_amount):
        """快三两面玩法分析"""
        two_sides_categories = ['两面']
        
        two_sides_group = group[group['玩法分类'].isin(two_sides_categories)]
        
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in two_sides_group.iterrows():
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
                '玩法分类': '两面',
                '矛盾类型': '、'.join(conflict_types),
                '投注内容': bet_content,
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面矛盾')
            }
            self._add_unique_result(results, '两面矛盾', record)

    # =============== 三色彩分析方法 ===============
    def analyze_three_color_patterns(self, df, period_amount_dict):
        """分析三色彩投注模式 - 添加金额参数"""
        results = defaultdict(list)
        
        df_target = df[df['彩种'].apply(self.identify_lottery_type) == 'THREE_COLOR']
        
        if len(df_target) == 0:
            return results
        
        grouped = df_target.groupby(['会员账号', '彩种', '期号'])
        
        for (account, lottery, period), group in grouped:
            # 获取该期投注金额
            period_key = f"{account}_{lottery}_{period}"
            period_amount = period_amount_dict.get(period_key, 0.0)
            
            self._analyze_three_color_zhengma(account, lottery, period, group, results, period_amount)
            self._analyze_three_color_two_sides(account, lottery, period, group, results, period_amount)
            self._analyze_three_color_wave(account, lottery, period, group, results, period_amount)
        
        return results
    
    def _analyze_three_color_zhengma(self, account, lottery, period, group, results, period_amount):
        zhengma_group = group[group['玩法分类'] == '正码']
        
        all_numbers = set()
        
        for _, row in zhengma_group.iterrows():
            content = str(row['内容'])
            numbers = self.data_analyzer.extract_numbers_from_content(content, 0, 9)
            all_numbers.update(numbers)
        
        if len(all_numbers) >= THRESHOLD_CONFIG['THREE_COLOR']['zhengma_multi']:
            record = {
                '会员账号': account,
                '彩种': lottery,
                '期号': period,
                '玩法分类': '正码',
                '号码数量': len(all_numbers),
                '投注内容': ', '.join([str(num) for num in sorted(all_numbers)]),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'号码数量': len(all_numbers)}, '正码多码')
            }
            self._add_unique_result(results, '正码多码', record)
    
    def _analyze_three_color_two_sides(self, account, lottery, period, group, results, period_amount):
        two_sides_group = group[group['玩法分类'] == '两面']
        
        has_big = False
        has_small = False
        has_single = False
        has_double = False
        
        for _, row in two_sides_group.iterrows():
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
                '玩法分类': '两面',
                '矛盾类型': '、'.join(conflict_types),
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'矛盾类型': '、'.join(conflict_types)}, '两面矛盾')
            }
            self._add_unique_result(results, '两面矛盾', record)
    
    def _analyze_three_color_wave(self, account, lottery, period, group, results, period_amount):
        """三色彩色波检测 - 记录同一期号内同时投注红波和绿波"""
        wave_group = group[group['玩法分类'] == '色波']
        
        # 收集该期号内所有波色投注
        all_waves = set()
        
        for _, row in wave_group.iterrows():
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
                '玩法分类': '色波',
                '投注波色数': len(all_waves),
                '投注波色': sorted(list(all_waves)),
                '投注内容': f"同一期号内投注: {', '.join(sorted(all_waves))}",
                '当期投注金额': period_amount,  # 添加金额信息
                '排序权重': self._calculate_sort_weight({'投注波色数': len(all_waves)}, '色波红绿投注')
            }
            self._add_unique_result(results, '色波红绿投注', record)
    
    def _calculate_sort_weight(self, record, result_type):
        """计算排序权重 - 优化版本"""
        weight = 0
        
        # 基于号码数量
        if record.get('号码数量', 0) > 0:
            weight += record['号码数量'] * 10
        
        # 基于矛盾类型复杂度
        if record.get('矛盾类型'):
            conflict_count = len(record['矛盾类型'].split('、'))
            weight += conflict_count * 15
        
        # 基于其他数量字段 - 优化：生肖数量、尾数数量等按照数量大小排序
        for field in ['生肖数量', '尾数数量', '投注区间数', '投注波色数', '投注五行数']:
            if record.get(field, 0) > 0:
                weight += record[field] * 8
        
        # 基于矛盾值 - 优化：和值变相超码按照相反方向的数量排序
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

    def _extract_exact_position_from_category(self, category):
        """从玩法分类中精确提取位置"""
        category_str = str(category).strip()
        
        # 精确的位置映射
        position_mapping = {
            '正码一': '正码一', '正1': '正码一', '正码1': '正码一',
            '正码二': '正码二', '正2': '正码二', '正码2': '正码二',
            '正码三': '正码三', '正3': '正码三', '正码3': '正码三',
            '正码四': '正码四', '正4': '正码四', '正码4': '正码四',
            '正码五': '正码五', '正5': '正码五', '正码5': '正码五',
            '正码六': '正码六', '正6': '正码六', '正码6': '正码六'
        }
        
        for key, position in position_mapping.items():
            if key in category_str:
                return position
        
        return '未知位置'
    
    def _normalize_zhengma_position_exact(self, position):
        """精确标准化正码位置名称"""
        exact_mapping = {
            '正码一': '正码一', '正1': '正码一', '正码1': '正码一', '1': '正码一',
            '正码二': '正码二', '正2': '正码二', '正码2': '正码二', '2': '正码二',
            '正码三': '正码三', '正3': '正码三', '正码3': '正码三', '3': '正码三',
            '正码四': '正码四', '正4': '正码四', '正码4': '正码四', '4': '正码四',
            '正码五': '正码五', '正5': '正码五', '正码5': '正码五', '5': '正码五',
            '正码六': '正码六', '正6': '正码六', '正码6': '正码六', '6': '正码六'
        }
        
        position = position.strip()
        if position in exact_mapping:
            return exact_mapping[position]
        
        # 模糊匹配
        for key, value in exact_mapping.items():
            if key in position:
                return value
        
        return '正码一'  # 默认返回正码一
    
    def _extract_wave_from_zhengma_content(self, content):
        """从正码内容中提取波色"""
        content_str = str(content).strip()
        waves = set()
        
        # 精确匹配波色关键词
        wave_keywords = {
            '红波': ['红波', '紅色波', '红'],
            '蓝波': ['蓝波', '藍波', '蓝', '藍'], 
            '绿波': ['绿波', '綠波', '绿', '綠']
        }
        
        for wave_name, keywords in wave_keywords.items():
            for keyword in keywords:
                # 精确匹配，避免部分匹配
                if keyword == content_str or f"-{keyword}" in content_str or f"{keyword}," in content_str:
                    waves.add(wave_name)
                    break
        
        return waves

    def _extract_exact_pk10_position_from_category(self, category):
        """从PK10玩法分类中精确提取位置 - 增强特殊空格处理"""
        category_str = str(category).strip()
        
        # 专门处理"1-5名"和"6-10名"这样的分类
        if category_str in ['1-5名', '1~5名', '1-5', '1~5']:
            # 返回一个列表或特殊标记
            return '1-5名'
        elif category_str in ['6-10名', '6~10名', '6-10', '6~10']:
            return '6-10名'
 
        # 专门处理冠军、亚军、季军的特殊空格问题
        # 将各种空格（包括全角空格、不间断空格等）统一替换为普通空格
        import re
        category_normalized = re.sub(r'[ \t\u00A0\u3000\u2000-\u200B]+', ' ', category_str)
        
        # 专门针对冠军、亚军、季军的精确匹配
        # 首先检查是否包含"龙虎_"前缀
        if '龙虎_' in category_normalized:
            # 提取"龙虎_"后面的部分
            position_part = category_normalized.split('龙虎_')[-1].strip()
            
            # 冠军的各种变体 - 增强匹配
            if any(keyword in position_part for keyword in ['冠军', '冠 军', '冠　军', '冠军', '冠  军']):
                return '冠军'
            # 亚军的各种变体 - 增强匹配
            elif any(keyword in position_part for keyword in ['亚军', '亚 军', '亚　军', '亚军', '亚  军']):
                return '亚军'
            # 季军的各种变体 - 增强匹配
            elif any(keyword in position_part for keyword in ['季军', '季 军', '季　军', '季军', '季  军']):
                return '季军'
            # 其他位置
            elif '第四名' in position_part:
                return '第四名'
            elif '第五名' in position_part:
                return '第五名'
            elif '第六名' in position_part:
                return '第六名'
            elif '第七名' in position_part:
                return '第七名'
            elif '第八名' in position_part:
                return '第八名'
            elif '第九名' in position_part:
                return '第九名'
            elif '第十名' in position_part:
                return '第十名'
        
        # 如果上述匹配失败，使用原有的精确位置映射
        position_mapping = {
            '冠军': ['冠军', '龙虎_冠军', '龙虎_冠 军', '冠 军', '前一'],
            '亚军': ['亚军', '龙虎_亚军', '龙虎_亚 军', '亚 军'],
            '季军': ['季军', '龙虎_季军', '龙虎_季 军', '季 军', '第三名'],
            '第四名': ['第四名', '龙虎_第四名', '第4名'],
            '第五名': ['第五名', '龙虎_第五名', '第5名'],
            '第六名': ['第六名', '龙虎_第六名', '第6名'],
            '第七名': ['第七名', '龙虎_第七名', '第7名'],
            '第八名': ['第八名', '龙虎_第八名', '第8名'],
            '第九名': ['第九名', '龙虎_第九名', '第9名'],
            '第十名': ['第十名', '龙虎_第十名', '第10名']
        }
        
        for position, keywords in position_mapping.items():
            for keyword in keywords:
                # 使用标准化后的分类进行匹配
                if keyword in category_normalized:
                    return position
        
        # 如果还是无法识别，尝试更宽松的匹配
        if any(word in category_normalized for word in ['冠军', '冠 军', '冠　军']):
            return '冠军'
        elif any(word in category_normalized for word in ['亚军', '亚 军', '亚　军']):
            return '亚军'
        elif any(word in category_normalized for word in ['季军', '季 军', '季　军']):
            return '季军'
        
        return '未知位置'

    def _analyze_detailed_category_patterns(self, account, lottery, period, group, results, 
                                          category_config, extract_method, count_field, 
                                          result_suffix, content_formatter=None):
        """
        通用详细分类检测方法
        category_config: 分类配置字典 {分类名: {阈值配置}}
        extract_method: 内容提取方法
        count_field: 数量字段名
        result_suffix: 结果后缀
        content_formatter: 内容格式化函数
        """
        for category_name, config in category_config.items():
            category_group = group[group['玩法分类'] == category_name]
            
            for _, row in category_group.iterrows():
                content = str(row['内容'])
                
                # 解析玩法-投注内容格式
                if '-' in content:
                    parts = content.split('-', 1)
                    bet_content = parts[1].strip()
                else:
                    bet_content = content
                    
                # 提取内容
                items = extract_method(bet_content)
                
                # 检测阈值
                if len(items) >= config['threshold']:
                    # 格式化显示内容
                    if content_formatter:
                        display_content = content_formatter(items)
                    else:
                        display_content = ', '.join(sorted([str(item) for item in items]))
                    
                    record = {
                        '会员账号': account,
                        '彩种': lottery,
                        '期号': period,
                        '玩法分类': f"{category_name}（{len(items)}{count_field}）",
                        '违规类型': f'{category_name}{result_suffix}',
                        count_field: len(items),
                        '投注内容': display_content,
                        '当期投注金额': period_amount,  # 添加金额信息
                        '排序权重': self._calculate_sort_weight({count_field: len(items)}, f'{category_name}{result_suffix}')
                    }
                    self._add_unique_result(results, f'{category_name}{result_suffix}', record)
    
    def analyze_all_patterns(self, df):
        """综合分析所有模式"""
        logger.info("开始综合分析所有彩票模式...")
        
        # 重置缓存
        self.seen_records = set()
        
        # 使用进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = {}
        # 修改这里：添加3D系列
        lottery_types = ['PK拾赛车', '时时彩', '六合彩', '快三', '三色彩', '3D系列']
        
        for i, lottery_type in enumerate(lottery_types):
            status_text.text(f"正在分析 {lottery_type}...")
            
            if lottery_type == 'PK拾赛车':
                all_results[lottery_type] = self.analyze_pk10_patterns(df)
            elif lottery_type == '时时彩':
                all_results[lottery_type] = self.analyze_ssc_patterns(df)
            elif lottery_type == '六合彩':
                all_results[lottery_type] = self.analyze_lhc_patterns(df)
            elif lottery_type == '快三':
                all_results[lottery_type] = self.analyze_k3_patterns(df)
            elif lottery_type == '三色彩':
                all_results[lottery_type] = self.analyze_three_color_patterns(df)
            # 添加3D系列分析
            elif lottery_type == '3D系列':
                all_results[lottery_type] = self.analyze_3d_patterns(df)
            
            progress_bar.progress((i + 1) / len(lottery_types))
        
        status_text.text("分析完成！")
        
        # 统计结果
        total_findings = 0
        for lottery_type, results in all_results.items():
            type_count = sum(len(records) for records in results.values())
            total_findings += type_count
            if type_count > 0:
                logger.info(f"{lottery_type}: 发现 {type_count} 条可疑记录")
                for result_type, records in results.items():
                    if records:
                        logger.info(f"  - {result_type}: {len(records)} 条")
        
        logger.info(f"总计发现 {total_findings} 条可疑记录")
        return all_results

# ==================== 结果处理器 ====================
class ResultProcessor:
    def __init__(self):
        self.behavior_names = {
            'PK拾赛车': {
                '冠军多码': '冠军多码',
                '亚军多码': '亚军多码',
                '第三名多码': '第三名多码',
                '第四名多码': '第四名多码',
                '第五名多码': '第五名多码',
                '第六名多码': '第六名多码',
                '第七名多码': '第七名多码',
                '第八名多码': '第八名多码',
                '第九名多码': '第九名多码',
                '第十名多码': '第十名多码',
                '超码': '超码',
                '冠亚和多码': '冠亚和多码',
                '冠亚和矛盾': '冠亚和矛盾',
                '两面矛盾': '两面矛盾',
                '独立玩法矛盾': '独立玩法矛盾',
                '前一多码': '前一多码',
                '龙虎矛盾': '龙虎矛盾',
                '多位置相同投注': '多位置相同投注',
                '十个位置相同投注': '十个位置相同投注'
            },
            '快三': {
                '和值多码': '和值多码',
                '和值矛盾': '和值矛盾',  # 大小单双同时下注
                '和值变相超码': '和值变相超码',  # 投注方向与号码分布矛盾
                '独胆多码': '独胆多码',
                '不同号全包': '不同号全包',
                '两面矛盾': '两面矛盾'
            },
            '六合彩': {
                '数字类多码': '数字类多码',
                '特码多码': '特码多码',
                '特码变相超码': '特码变相超码',
                '特码区间变相超码': '特码区间变相超码',
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
                '正1特多码': '正1特多码',
                '正2特多码': '正2特多码', 
                '正3特多码': '正3特多码',
                '正4特多码': '正4特多码',
                '正5特多码': '正5特多码',
                '正6特多码': '正6特多码',
                '正1特矛盾': '正1特矛盾',
                '正2特矛盾': '正2特矛盾',
                '正3特矛盾': '正3特矛盾',
                '正4特矛盾': '正4特矛盾',
                '正5特矛盾': '正5特矛盾',
                '正6特矛盾': '正6特矛盾',
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
                '正码一波色全包': '正码一波色全包',
                '正码二波色全包': '正码二波色全包',
                '正码三波色全包': '正码三波色全包',
                '正码四波色全包': '正码四波色全包',
                '正码五波色全包': '正码五波色全包',
                '正码六波色全包': '正码六波色全包',        
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
                '百位多码': '百位多码',
                '十位多码': '十位多码',
                '个位多码': '个位多码',
                '两面矛盾': '两面矛盾',
                '定位胆多码': '定位胆多码'
            },
            '时时彩': {
                '第1球多码': '第1球多码',
                '第2球多码': '第2球多码',
                '第3球多码': '第3球多码',
                '第4球多码': '第4球多码',
                '第5球多码': '第5球多码',
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
        """组织结果按账户分类 - 添加金额信息"""
        account_results = defaultdict(lambda: {
            'violations': [],
            'periods': set(),
            'violation_types': set(),
            'violation_count': 0,
            'lottery_types': set(),
            'violations_by_type': defaultdict(list),
            'violations_by_lottery': defaultdict(lambda: defaultdict(list)),
            'total_bet_amount': 0.0,  # 添加总投注金额
            'period_amounts': defaultdict(float)  # 添加每期金额
        })
        
        for lottery_type, results in all_results.items():
            for result_type, records in results.items():
                for record in records:
                    account = record['会员账号']
                    period = record['期号']
                    lottery = record['彩种']
                    amount = record.get('当期投注金额', 0.0)
                    
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
                        '当期投注金额': amount,  # 添加金额信息
                        '排序权重': record.get('排序权重', 0)
                    }
                    
                    account_results[account]['violations'].append(violation_record)
                    account_results[account]['violations_by_type'][result_type].append(violation_record)
                    account_results[account]['violations_by_lottery'][lottery][result_type].append(violation_record)
                    account_results[account]['periods'].add(period)
                    account_results[account]['violation_types'].add(result_type)
                    account_results[account]['violation_count'] += 1
                    account_results[account]['lottery_types'].add(lottery)
                    account_results[account]['total_bet_amount'] += amount
                    account_results[account]['period_amounts'][period] = amount
        
        return account_results
    
    def _get_violation_details(self, record, result_type):
        """获取违规详情 - 增强分类显示"""
        details = []
        
        # 专门处理龙虎矛盾的显示
        if '龙虎矛盾' in result_type:
            if record.get('位置'):
                details.append(f"位置: {record['位置']}")
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            return ' | '.join(details) if details else '无详情'
        
        # 专门处理独立玩法矛盾的显示
        if '独立玩法矛盾' in result_type:
            if record.get('位置'):
                details.append(f"位置: {record['位置']}")
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            return ' | '.join(details) if details else '无详情'
        
        # 专门处理和值变相超码的显示
        if '和值变相超码' in result_type:
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            if record.get('矛盾值', 0) > 0:
                details.append(f"矛盾值: {record['矛盾值']}")
            return ' | '.join(details) if details else '无详情'
        
        # 专门处理和值矛盾的显示
        elif '和值矛盾' in result_type:
            if record.get('矛盾类型'):
                details.append(f"矛盾类型: {record['矛盾类型']}")
            return ' | '.join(details) if details else '无详情'
        
        # 专门处理多位置相同投注的显示（新增的部分）
        elif result_type in ['多位置相同投注', '十个位置相同投注']:
            # 投注类型和内容
            if record.get('投注类型'):
                if record.get('投注项'):
                    if record['投注类型'] == '号码':
                        details.append(f"号码: {record['投注项']}")
                    else:
                        details.append(f"{record['投注类型']}: {record['投注项']}")
            
            # 位置数量
            if record.get('位置数量', 0) > 0:
                details.append(f"位置数量: {record['位置数量']}")
            
            # 出现位置
            if record.get('出现位置'):
                details.append(f"出现位置: {record['出现位置']}")
            
            return ' | '.join(details) if details else '无详情'
        
        # 尾数多码的特殊处理
        elif '尾数' in result_type:
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
        """优化显示记录 - 增强去重逻辑"""
        if not records:
            return []
        
        # 重置缓存（每次调用时重新计算）
        self.displayed_records_cache = set()
        
        def get_record_key(record):
            """生成记录的唯一键 - 增强版本"""
            return (
                record.get('会员账号', ''),
                record.get('期号', ''),
                record.get('玩法分类', ''),
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
        
        # 对于龙虎矛盾，确保优先显示
        dragon_tiger_records = [r for r in unique_records if '龙虎矛盾' in r.get('违规类型', '')]
        other_records = [r for r in unique_records if '龙虎矛盾' not in r.get('违规类型', '')]
        
        # 优先显示龙虎矛盾记录
        result_records = dragon_tiger_records + other_records
        
        return result_records[:max_records]
    
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
        """创建汇总统计 - 修改版本：只记录违规彩种"""
        total_violations = sum(data['violation_count'] for data in account_results.values())
        
        summary = {
            '总记录数': len(df_clean),
            '总会员数': df_clean['会员账号'].nunique(),
            '彩种数量': df_clean['彩种'].nunique(),
            '违规账户数': len(account_results),
            '总违规记录数': total_violations,
            '违规类型统计': defaultdict(int),
            '账户违规统计': [],
            '账户违规彩种详情': defaultdict(list)  # 修改：只存储违规彩种详情
        }
        
        for account, data in account_results.items():
            for violation_type in data['violation_types']:
                summary['违规类型统计'][violation_type] += len(data['violations_by_type'][violation_type])
            
            # 计算该账户在所有彩种的总投注期数
            total_periods = 0
            lottery_count = 0
            
            # 收集违规彩种详情
            violation_lotteries = set()
            violation_periods = {}
            
            # 遍历该账户涉及的所有违规彩种
            for lottery, lottery_data in data['violations_by_lottery'].items():
                if lottery_data:  # 如果该彩种有违规记录
                    violation_lotteries.add(lottery)
                    # 计算该彩种的投注期数
                    periods_count = df_clean[(df_clean['会员账号'] == account) & 
                                           (df_clean['彩种'] == lottery)]['期号'].nunique()
                    violation_periods[lottery] = periods_count
                    total_periods += periods_count
                    lottery_count += 1
            
            # 存储违规彩种详情
            lottery_details = []
            for lottery in sorted(violation_lotteries):
                periods = violation_periods.get(lottery, 0)
                lottery_details.append(f"{lottery}({periods}期)")
            
            summary['账户违规彩种详情'][account] = lottery_details
            
            summary['账户违规统计'].append({
                '账户': account,
                '彩种投注期数': total_periods,  # 只计算违规彩种的注期数
                '违规次数': data['violation_count'],
                '违规类型数': len(data['violation_types']),
                '彩种数': len(violation_lotteries)  # 只计算违规彩种数
            })
        
        summary['账户违规统计'] = sorted(summary['账户违规统计'], key=lambda x: x['违规次数'], reverse=True)
        
        return summary
    
    def display_summary(self, summary):
        """显示汇总统计 - 修改版本：只显示违规彩种"""
        st.subheader("📊 汇总统计")
        
        # 五个指标：总记录数、总会员数、彩种数量、违规账户数、总违规记录数
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("总记录数", summary['总记录数'])
        with col2:
            st.metric("总会员数", summary['总会员数'])
        with col3:
            st.metric("彩种数量", summary['彩种数量'])
        with col4:
            st.metric("违规账户数", summary['违规账户数'])
        with col5:
            st.metric("总违规记录数", summary['总违规记录数'])
        
        if summary['账户违规统计']:
            with st.expander("👥 参与账户详细统计", expanded=True):  # 默认打开
                # 创建新的DataFrame显示格式
                account_stats = []
                for account_stat in summary['账户违规统计']:
                    account = account_stat['账户']
                    account_stats.append({
                        '账户': account,
                        '彩种投注期数': account_stat['彩种投注期数'],
                        '违规次数': account_stat['违规次数'],
                        '违规类型数': account_stat['违规类型数'],
                        '彩种数': account_stat['彩种数'],
                        '违规彩种': '; '.join(summary['账户违规彩种详情'].get(account, []))  # 修改：只显示违规彩种
                    })
                
                account_df = pd.DataFrame(account_stats)
                
                # 使用更友好的列名显示
                display_df = account_df.rename(columns={
                    '账户': '会员账号',
                    '彩种投注期数': '彩种投注期数',
                    '违规次数': '违规次数',
                    '违规类型数': '违规类型数',
                    '彩种数': '涉及彩种数',
                    '违规彩种': '违规彩种（彩种(期数)）'  # 修改：只显示违规彩种
                })
                
                st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    def display_account_results(self, account_results):
        """显示账户结果 - 添加金额显示"""
        if not account_results:
            st.info("🎉 未发现可疑投注行为")
            return
        
        st.subheader("🔍 违规账户详情")
        
        sorted_accounts = sorted(account_results.items(), 
                               key=lambda x: x[1]['total_bet_amount'],  # 按总投注金额排序
                               reverse=True)
        
        for account_index, (account, data) in enumerate(sorted_accounts, 1):
            # 转义账号中的下划线
            account_display = account.replace('_', '\\_')
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])  # 增加一列显示金额
                
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
                
                with col4:
                    # 显示总投注金额
                    st.write(f"**总投注金额:** {data['total_bet_amount']:,.2f}")
                
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
                                
                                # 准备显示数据 - 添加金额列
                                display_data = []
                                for record in representative_records:
                                    display_record = {
                                        '期号': record['期号'],
                                        '玩法分类': record['玩法分类'],
                                        '违规类型': violation_type,
                                        '详细信息': record.get('详细信息', ''),
                                        '投注内容': record.get('投注内容', ''),
                                        '投注金额': f"{record.get('当期投注金额', 0):,.2f}"  # 添加金额列
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
        """准备导出数据 - 增强版本，包含所有字段和金额"""
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
                            '行为类型': behavior_type,
                            '违规类型': record.get('违规类型', ''),
                            '投注内容': record.get('投注内容', ''),  # 添加投注内容
                            '投注项': record.get('投注项', ''),      # 添加投注项
                            '投注类型': record.get('投注类型', ''),  # 添加投注类型
                            '位置数量': record.get('位置数量', 0),   # 添加位置数量
                            '出现位置': record.get('出现位置', ''),  # 添加出现位置
                            '当期投注金额': record.get('当期投注金额', 0.0),  # 添加金额
                            '详细信息': record.get('详细信息', '无详情')  # 添加详细信息
                        }
                        
                        # 添加矛盾类型
                        if '矛盾类型' in record:
                            export_record['矛盾类型'] = record['矛盾类型']
                        
                        # 添加数量信息
                        self._add_quantity_info(export_record, record, behavior_type)
                        
                        # 添加其他字段
                        for field in ['号码数量', '生肖数量', '尾数数量', '投注区间数', 
                                     '投注波色数', '投注五行数', '投注半波数']:
                            if field in record:
                                export_record[field] = record[field]
                        
                        export_data.append(export_record)
        
        return export_data
    
    def _add_quantity_info(self, export_record, record, behavior_type):
        """添加数量信息到导出记录 - 增强版本"""
        quantity_fields = {
            # 快三相关
            '和值多码': ('号码数量', '投注内容'),
            '和值矛盾': (None, '投注内容'),  # 和值矛盾只有投注内容
            '和值变相超码': ('矛盾值', '投注内容'),  # 和值变相超码有矛盾值
            '独胆多码': ('号码数量', '投注内容'),
            '不同号全包': ('号码数量', '投注内容'),
            '两面矛盾': (None, '投注内容'),
            
            # 多位置相同投注相关
            '多位置相同投注': ('位置数量', '投注内容'),
            '十个位置相同投注': ('位置数量', '投注内容'),
            
            # 六合彩相关 - 基础检测
            '数字类多码': ('号码数量', '投注内容'),
            '特码多码': ('号码数量', '投注内容'),
            '正码多码': ('号码数量', '投注内容'),
            '正码1-6多码': ('号码数量', '投注内容'),
            '正特多码': ('号码数量', '投注内容'),
            
            # 六合彩生肖类
            '生肖类多码': ('生肖数量', '投注内容'),
            '平特多肖': ('生肖数量', '投注内容'),
            '特肖多肖': ('生肖数量', '投注内容'),
            '一肖多肖': ('生肖数量', '投注内容'),
            
            # 六合彩尾数类
            '尾数多码': ('尾数数量', '投注内容'),
            '尾数头尾多码': ('尾数数量', '投注内容'),
            '特尾多尾': ('尾数数量', '投注内容'),
            '全尾多尾': ('尾数数量', '投注内容'),
            
            # 六合彩连肖连尾
            '连肖多肖': ('生肖数量', '投注内容'),
            '连尾多尾': ('尾数数量', '投注内容'),
            
            # 六合彩区间波色五行
            '区间多组': ('投注区间数', '投注内容'),
            '波色三组': ('投注波色数', '投注内容'),
            '色波三组': ('投注波色数', '投注内容'),
            '色波全包': ('投注波色数', '投注内容'),
            '半波单双全包': ('投注半波数', '投注内容'),
            '半波大小全包': ('投注半波数', '投注内容'),
            '五行多组': ('投注五行数', '投注内容'),
            
            # 六合彩矛盾检测
            '两面玩法矛盾': (None, '投注内容'),
            '正码1-6矛盾': (None, '投注内容'),
            
            # 六合彩正码波色全包
            '正码一波色全包': ('投注波色数', '投注内容'),
            '正码二波色全包': ('投注波色数', '投注内容'),
            '正码三波色全包': ('投注波色数', '投注内容'),
            '正码四波色全包': ('投注波色数', '投注内容'),
            '正码五波色全包': ('投注波色数', '投注内容'),
            '正码六波色全包': ('投注波色数', '投注内容'),
            
            # 六合彩正特矛盾
            '正特矛盾': (None, '投注内容'),
            
            # 六合彩正特具体位置
            '正1特多码': ('号码数量', '投注内容'),
            '正2特多码': ('号码数量', '投注内容'),
            '正3特多码': ('号码数量', '投注内容'),
            '正4特多码': ('号码数量', '投注内容'),
            '正5特多码': ('号码数量', '投注内容'),
            '正6特多码': ('号码数量', '投注内容'),
            '正1特矛盾': (None, '投注内容'),
            '正2特矛盾': (None, '投注内容'),
            '正3特矛盾': (None, '投注内容'),
            '正4特矛盾': (None, '投注内容'),
            '正5特矛盾': (None, '投注内容'),
            '正6特矛盾': (None, '投注内容'),
            
            # 六合彩半波相关
            '半波全包': (None, '投注内容'),
            '半波多组投注': ('投注波色数', '投注内容'),
            
            # 六合彩特码变相超码（新增）
            '特码变相超码': ('矛盾值', '投注内容'),
            '特码区间变相超码': ('矛盾值', '投注内容'),
            
            # 三色彩相关
            '色波全包': ('投注波色数', '投注内容'),
            '色波红绿投注': ('投注波色数', '投注内容'),
            
            # 3D系列相关
            '百位多码': ('号码数量', '投注内容'),
            '十位多码': ('号码数量', '投注内容'),
            '个位多码': ('号码数量', '投注内容'),
            '两面矛盾': (None, '投注内容'),
            '定位胆多码': ('号码数量', '投注内容'),
            
            # 时时彩相关
            '斗牛多码': ('号码数量', '投注内容'),
            '定位胆多码': ('号码数量', '投注内容'),
            '第1球多码': ('号码数量', '投注内容'),
            '第2球多码': ('号码数量', '投注内容'),
            '第3球多码': ('号码数量', '投注内容'),
            '第4球多码': ('号码数量', '投注内容'),
            '第5球多码': ('号码数量', '投注内容'),
            
            # PK10相关
            '超码': ('号码数量', '投注内容'),
            '冠军多码': ('号码数量', '投注内容'),
            '亚军多码': ('号码数量', '投注内容'),
            '第三名多码': ('号码数量', '投注内容'),
            '第四名多码': ('号码数量', '投注内容'),
            '第五名多码': ('号码数量', '投注内容'),
            '第六名多码': ('号码数量', '投注内容'),
            '第七名多码': ('号码数量', '投注内容'),
            '第八名多码': ('号码数量', '投注内容'),
            '第九名多码': ('号码数量', '投注内容'),
            '第十名多码': ('号码数量', '投注内容'),
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
            
            # 添加数量字段
            if count_field and count_field in record:
                export_record[count_field] = record[count_field]
            
            # 添加投注内容字段
            if content_field:
                # 优先使用record中的投注内容，如果没有则尝试构建
                if record.get(content_field):
                    export_record[content_field] = str(record[content_field])
                elif record.get('投注内容'):
                    export_record[content_field] = str(record['投注内容'])
                elif behavior_type in ['多位置相同投注', '十个位置相同投注']:
                    # 对于多位置相同投注，构建投注内容
                    if record.get('投注类型') and record.get('投注项'):
                        if record['投注类型'] == '号码':
                            export_record[content_field] = f"号码{record['投注项']}"
                        else:
                            export_record[content_field] = record['投注项']
        
        # 添加位置信息（3D系列、PK10、时时彩等位置投注专用）
        if record.get('位置'):
            export_record['位置'] = record['位置']
        
        # 确保投注内容字段一定存在
        if '投注内容' not in export_record or not export_record['投注内容']:
            if record.get('投注内容'):
                export_record['投注内容'] = record['投注内容']
            elif record.get('投注项'):
                if record.get('投注类型') == '号码':
                    export_record['投注内容'] = f"号码{record['投注项']}"
                else:
                    export_record['投注内容'] = record['投注项']
        
        # 添加特码变相超码特有的数量字段
        if behavior_type == '特码变相超码':
            for field in ['小号码数量', '大号码数量', '单号码数量', '双号码数量']:
                if field in record:
                    export_record[field] = record[field]
        
        # 添加和值变相超码特有的数量字段
        if behavior_type == '和值变相超码':
            for field in ['小号码数量', '大号码数量', '单号码数量', '双号码数量']:
                if field in record:
                    export_record[field] = record[field]
        
        # 添加通用的数量字段
        for field in ['号码数量', '生肖数量', '尾数数量', '投注区间数', 
                     '投注波色数', '投注五行数', '投注半波数', '位置数量']:
            if field in record:
                export_record[field] = record[field]
        
        # 添加矛盾类型信息（如果存在）
        if record.get('矛盾类型'):
            export_record['矛盾类型'] = record['矛盾类型']
        
        # 添加投注项和投注类型（多位置相同投注等）
        if record.get('投注项'):
            export_record['投注项'] = record['投注项']
        if record.get('投注类型'):
            export_record['投注类型'] = record['投注类型']
        
        # 添加出现位置信息（多位置相同投注）
        if record.get('出现位置'):
            export_record['出现位置'] = record['出现位置']
    
    def export_to_excel(self, account_summary, filename_prefix="彩票分析结果"):
        """导出分析结果到Excel文件"""
        try:
            export_data = self.prepare_export_data(account_summary)
            
            if not export_data:
                st.warning("没有可导出的数据")
                return
            
            # 创建DataFrame
            df_export = pd.DataFrame(export_data)
   
            # 确保投注内容列存在
            if '投注内容' not in df_export.columns:
                st.warning("警告：导出数据中缺少'投注内容'列")
                # 尝试从其他字段构建投注内容
                if '投注项' in df_export.columns and '投注类型' in df_export.columns:
                    def build_bet_content(row):
                        if row['投注类型'] == '号码' and pd.notna(row['投注项']):
                            return f"号码{row['投注项']}"
                        elif pd.notna(row['投注项']):
                            return row['投注项']
                        return ''
                    
                    df_export['投注内容'] = df_export.apply(build_bet_content, axis=1)
            
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
            
            # 显示导出数据的预览
            with st.expander("📋 导出数据预览", expanded=False):
                st.dataframe(df_export.head(10))
            
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
            import traceback
            logger.error(f"导出错误详情: {traceback.format_exc()}")
    
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
        pk10_same_number = st.slider("同号多位置阈值", 3, 10, 7)  # 新增
        THRESHOLD_CONFIG['PK10']['multi_number'] = pk10_multi
        THRESHOLD_CONFIG['PK10']['gyh_multi_number'] = pk10_gyh
        THRESHOLD_CONFIG['PK10']['same_number_multi_position'] = pk10_same_number  # 新增
    
    with st.sidebar.expander("时时彩系列阈值"):
        ssc_dingwei = st.slider("定位胆多码阈值", 5, 15, THRESHOLD_CONFIG['SSC']['dingwei_multi'])
        ssc_douniu = st.slider("斗牛多码阈值", 5, 15, THRESHOLD_CONFIG['SSC']['douniu_multi'])
        THRESHOLD_CONFIG['SSC']['dingwei_multi'] = ssc_dingwei
        THRESHOLD_CONFIG['SSC']['douniu_multi'] = ssc_douniu
    
    with st.sidebar.expander("六合彩系列阈值"):
        lhc_numbers = st.slider("数字类多码阈值", 20, 50, THRESHOLD_CONFIG['LHC']['number_play'])
        lhc_zodiacs = st.slider("生肖类多码阈值", 5, 15, THRESHOLD_CONFIG['LHC']['zodiac_play'])
        lhc_tails = st.slider("尾数多码阈值", 5, 15, THRESHOLD_CONFIG['LHC']['tail_play'])
        lhc_contradiction = st.slider("特码变相超码阈值", 3, 20, 7)  # 新增：特码变相超码阈值
        
        THRESHOLD_CONFIG['LHC']['number_play'] = lhc_numbers
        THRESHOLD_CONFIG['LHC']['zodiac_play'] = lhc_zodiacs
        THRESHOLD_CONFIG['LHC']['tail_play'] = lhc_tails
        THRESHOLD_CONFIG['LHC']['value_size_contradiction'] = lhc_contradiction  # 新增
    
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
                analyzer = AnalysisEngine()
                result_processor = ResultProcessor()
                exporter = Exporter()
                
                # 数据清洗
                df_clean = processor.clean_data(uploaded_file)
                
                if df_clean is not None and len(df_clean) > 0:

                    # 统一玩法分类
                    df_normalized = analyzer.normalize_play_categories(df_clean)
                    
                    # 创建期号金额字典
                    period_amount_dict = {}
                    if '金额' in df_normalized.columns:
                        # 确保金额列是数值类型
                        if df_normalized['金额'].dtype != 'float64':
                            df_normalized['金额'] = pd.to_numeric(df_normalized['金额'], errors='coerce')
                        
                        # 按会员账号、彩种、期号分组，计算每期总金额
                        for (account, lottery, period), group in df_normalized.groupby(['会员账号', '彩种', '期号']):
                            period_key = f"{account}_{lottery}_{period}"
                            total_amount = group['金额'].sum()
                            period_amount_dict[period_key] = total_amount
                    
                    # 分析投注模式
                    # 使用进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    all_results = {}
                    # 明确定义 lottery_types 变量 - 添加3D系列
                    lottery_types = ['PK拾赛车', '时时彩', '六合彩', '快三', '三色彩', '3D系列']
                    
                    for i, lottery_type in enumerate(lottery_types):
                        status_text.text(f"正在分析 {lottery_type}...")
                        
                        if lottery_type == 'PK拾赛车':
                            all_results[lottery_type] = analyzer.analyze_pk10_patterns(df_normalized, period_amount_dict)
                        elif lottery_type == '时时彩':
                            all_results[lottery_type] = analyzer.analyze_ssc_patterns(df_normalized, period_amount_dict)
                        elif lottery_type == '六合彩':
                            all_results[lottery_type] = analyzer.analyze_lhc_patterns(df_normalized, period_amount_dict)
                        elif lottery_type == '快三':
                            all_results[lottery_type] = analyzer.analyze_k3_patterns(df_normalized, period_amount_dict)
                        elif lottery_type == '三色彩':
                            all_results[lottery_type] = analyzer.analyze_three_color_patterns(df_normalized, period_amount_dict)
                        # 添加3D系列分析调用
                        elif lottery_type == '3D系列':
                            all_results[lottery_type] = analyzer.analyze_3d_patterns(df_normalized, period_amount_dict)
                        
                        progress_bar.progress((i + 1) / len(lottery_types))
                    
                    status_text.text("分析完成！")
                    
                    # 统计结果
                    total_findings = 0
                    for lottery_type, results in all_results.items():
                        type_count = sum(len(records) for records in results.values())
                        total_findings += type_count

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
            import traceback
            st.error(f"详细错误信息: {traceback.format_exc()}")
            logger.error(f"处理过程中出现错误: {str(e)}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")
    
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
        - ✅ 快三系列：和值多码、和值矛盾、和值变相超码、独胆多码、不同号全包、两面矛盾
        - ✅ 三色彩系列：正码多码、两面矛盾、色波矛盾
        - ✅ 3D系列：定位胆多码、两面矛盾
        
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
