import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def getCorelationColumns(corr_matrix) -> set:
    threshold = 0.8
    correlated_pairs = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlated_pairs.add((col1, col2))

    return correlated_pairs


def main():
    data = pd.read_csv("AmesHousing.csv")

    label_encoder = LabelEncoder()
    columns = data.select_dtypes(exclude=['number'])

    for colum in columns:
        data[colum] = label_encoder.fit_transform(data[colum].astype(str))

    corr_matrix = data.corr()

    corr_columns = getCorelationColumns(corr_matrix)

    columns_to_delete = [i[0] for i in corr_columns]

    data.drop(columns=columns_to_delete)

    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    simpleImputer = SimpleImputer(strategy='mean')
    X = simpleImputer.fit_transform(X)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    scatter = ax.scatter(
        X_pca[:, 0],  # x - первый главный компонент
        X_pca[:, 1],  # y - второй главный компонент
        y,  # z - целевое значение (SalePrice)
        c=y,  # Цвет точек зависит от SalePrice
    )

    ax.set_xlabel("Главная компонента 1 (PCA)")
    ax.set_ylabel("Главная компонента 2 (PCA)")
    ax.set_zlabel("Целевое значение (SalePrice)")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("SalePrice")

    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alphas = np.logspace(-2, 3, 100) # 0.001 до 1000.

    lassoErrors = []

    for alpha in alphas:
        lassoModel = Lasso(alpha=alpha, max_iter=10000)
        lassoModel.fit(x_train, y_train)

        yPredLasso = lassoModel.predict(x_test)

        rmse = mean_squared_error(y_test, yPredLasso)

        lassoErrors.append(rmse)

    optimalAlpha = alphas[np.argmin(lassoErrors)]
    print(f"Оптимальное значение alpha: {optimalAlpha}")
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, lassoErrors, label="Lasso", color="brown")
    plt.xscale("log")
    plt.xlabel("Alpha (коэффициент регуляризации)")
    plt.ylabel("RMSE")
    plt.title("Зависимость ошибки от коэффициента регуляризации (Lasso)")
    plt.axvline(optimalAlpha, color="red", linestyle="--", label=f"Оптимальное alpha: {optimalAlpha:.3f}")
    plt.legend()
    plt.show()

    lassoModel = Lasso(alpha=optimalAlpha, max_iter=10000)
    lassoModel.fit(x_train, y_train)


    feature_names = data.drop(columns=["SalePrice"]).columns
    coefficients = lassoModel.coef_

    coefDataFrame = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    coefDataFrame['Absolute Coefficient'] = coefDataFrame['Coefficient'].abs()
    coefDataFrame = coefDataFrame.sort_values(by='Absolute Coefficient', ascending=False)


    mostInfluentialFeature = coefDataFrame.iloc[0]
    print(f"Наиболее влиятельный признак: {mostInfluentialFeature['Feature']}")
    print(f"Коэффициент: {mostInfluentialFeature['Coefficient']}")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=coefDataFrame.head(10))
    plt.title("Топ-10 наиболее влиятельных признаков")
    plt.show()


if __name__ == "__main__":
    main()
