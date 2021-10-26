from dateutil.relativedelta import relativedelta
from datetime import datetime
BASE_PATH = "./dashboard/data"

COLUMNS = [
        "id",
        "symbol",
        "name",
        "market_cap",
        "market_cap_rank",
        "current_price",
        "total_volume",
        "circulating_supply",
        "max_supply"
]

TODAY = datetime.now()
TODAY_STR = TODAY.strftime("%Y-%m-%d %H:%M:%S")

INIT_DATE = TODAY - relativedelta(years=2)

INIT_DATE = datetime(INIT_DATE.year, INIT_DATE.month, INIT_DATE.day)