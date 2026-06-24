from futu import *

quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

ret, data = quote_ctx.get_market_snapshot(["US.SPY", "US.QQQ", "US.TLT"])

if ret == RET_OK:
    print(data)
else:
    print("Error:", data)

quote_ctx.close()