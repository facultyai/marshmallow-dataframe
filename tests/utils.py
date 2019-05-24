def serialize_df(df, orient="split"):
    test_df = df.copy()
    if "datetime" in test_df.columns:
        # convert all datetimes to strings to enforce validation
        test_df["datetime"] = test_df["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    if orient == "records":
        return {"data": test_df.to_dict(orient="records")}
    elif orient == "split":
        if test_df.index.dtype.kind == "M":
            test_df.index = test_df.index.strftime("%Y-%m-%d %H:%M:%S")
        return test_df.to_dict(orient="split")
