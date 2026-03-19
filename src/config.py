"""
Project Babel — 日米業種リードラグ投資戦略
設定ファイル: 銘柄リスト・パラメータ・分類ラベル
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# --- 日本 TOPIX-17業種ETF ---
JP_TICKERS = [
    "1617.T",  # 食品
    "1618.T",  # エネルギー資源
    "1619.T",  # 建設・資材
    "1620.T",  # 素材・化学
    "1621.T",  # 医薬品
    "1622.T",  # 自動車・輸送機
    "1623.T",  # 鉄鋼・非鉄
    "1624.T",  # 機械
    "1625.T",  # 電機・精密
    "1626.T",  # 情報通信・サービスその他
    "1627.T",  # 電力・ガス
    "1628.T",  # 運輸・物流
    "1629.T",  # 商社・卸売
    "1630.T",  # 小売
    "1631.T",  # 銀行
    "1632.T",  # 金融（除く銀行）
    "1633.T",  # 不動産
]

JP_TICKER_NAMES = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}

# --- 米国 Select Sector SPDR ETF ---
US_TICKERS = [
    "XLB",   # 素材
    "XLC",   # コミュニケーション (2018/6/18〜)
    "XLE",   # エネルギー
    "XLF",   # 金融
    "XLI",   # 資本財
    "XLK",   # テクノロジー
    "XLP",   # 生活必需品
    "XLRE",  # 不動産 (2015/10/7〜)
    "XLU",   # 公益
    "XLV",   # ヘルスケア
    "XLY",   # 一般消費財
]

US_TICKER_NAMES = {
    "XLB": "素材",
    "XLC": "コミュニケーション",
    "XLE": "エネルギー",
    "XLF": "金融",
    "XLI": "資本財",
    "XLK": "テクノロジー",
    "XLP": "生活必需品",
    "XLRE": "不動産",
    "XLU": "公益",
    "XLV": "ヘルスケア",
    "XLY": "一般消費財",
}

# 途中上場銘柄
US_LATE_LISTING = {
    "XLC": "2018-06-18",
    "XLRE": "2015-10-07",
}

# --- シクリカル/ディフェンシブ分類 ---
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
US_NEUTRAL = [t for t in US_TICKERS if t not in US_CYCLICAL and t not in US_DEFENSIVE]

JP_CYCLICAL = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]
JP_NEUTRAL = [t for t in JP_TICKERS if t not in JP_CYCLICAL and t not in JP_DEFENSIVE]

# --- ハイパーパラメータ ---
ROLLING_WINDOW = 60        # ローリングウィンドウ L=60日
REGULARIZATION_LAMBDA = 0.9  # 正則化パラメータ λ=0.9
NUM_FACTORS = 3            # 主成分数 K=3
QUANTILE_THRESHOLD = 0.3   # ロング/ショート選定閾値 q=0.3

# --- 期間設定 ---
DATA_START = "2010-01-01"
DATA_END = None  # None = 直近まで
CFULL_START = "2010-01-01"
CFULL_END = "2014-12-31"
TEST_START = "2015-01-01"
TEST_END = None  # None = 直近まで

# --- Fama-French Data Library ---
FF_FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_Daily_CSV.zip"
FF_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_Mom_Factor_Daily_CSV.zip"
# 日本市場用
FF_JAPAN_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_3_Factors_Daily_CSV.zip"
FF_JAPAN_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_Mom_Factor_Daily_CSV.zip"
