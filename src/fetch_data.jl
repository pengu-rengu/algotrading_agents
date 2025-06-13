using HTTP

const api_key = "ASTr9QDuAiOrUygKF0nAiPDZp7PHWDIA"
resp = HTTP.get("https://financialmodelingprep.com/stable/historical-price-eod/light?symbol=BTCUSD&apikey=$(api_key)")

open(joinpath(@__DIR__, "response.json"), "w") do file
    write(file, String(resp.body))
end