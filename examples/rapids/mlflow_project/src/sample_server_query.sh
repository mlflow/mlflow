curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["Year", "Month", "DayofMonth", "DayofWeek", "CRSDepTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "ActualElapsedTime", "Origin"    , "Dest", "Distance", "Diverted"],"data":[[1987, 10, 1, 4, 1, 556, 0, 190, 247, 202, 162, 1846, 0]]}' http://127.0.0.1:55756/invocations

