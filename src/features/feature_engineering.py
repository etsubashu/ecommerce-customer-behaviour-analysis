def create_features(df):

    df["Spending_Per_Age"] = (
        df["Purchase_Amount"] / df["Age"]
    )

    return df