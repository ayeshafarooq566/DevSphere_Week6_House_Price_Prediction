from sklearn.datasets import fetch_california_housing

print("Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print("\nDATASET LOADED SUCCESSFULLY!")

print("\nDataset shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature names:")
print(list(X.columns))

print("\nFirst 5 rows:")
print(X.head())

print("\nTarget values:")
print(y.head())

print("\nMissing values:")
print(X.isnull().sum())

print("\nDataset inspection completed!")