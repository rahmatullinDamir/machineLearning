import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv("AmesHousing.csv")

    label_encoder = LabelEncoder()
    columns = data.select_dtypes(exclude=['number'])

    for colum in columns:
        data[colum] = label_encoder.fit_transform(data[colum].astype(str))

    corr_matrix = data.corr().abs()
    threshold = 0.8

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    corr_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    data = data.drop(columns=corr_columns)

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

    ax.set_xlabel("Первая компонента PCA")
    ax.set_ylabel("Вторая компонента PCA")
    ax.set_zlabel("SalePrice")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("SalePrice")

    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alphas = np.logspace(-2, 3, 100)  # 0.001 до 1000.

    erorrs = []

    for alpha in alphas:
        lassoModel = Lasso(alpha=alpha, max_iter=10000)
        lassoModel.fit(x_train, y_train)

        yPredLasso = lassoModel.predict(x_test)

        rmse = mean_squared_error(y_test, yPredLasso)

        erorrs.append(rmse)

    best_alpha = alphas[np.argmin(erorrs)]
    print(f"Оптимальное значение alpha: {best_alpha}")
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, erorrs, label="Lasso", color="brown")
    plt.xscale("log")
    plt.xlabel("Alpha (коэффициент регуляризации)")
    plt.ylabel("RMSE")
    plt.title("Зависимость ошибки от коэффициента регуляризации (Lasso)")
    plt.axvline(best_alpha, color="red", linestyle="--", label=f"Оптимальное alpha: {best_alpha:.3f}")
    plt.legend()
    plt.show()

    lassoModel = Lasso(alpha=best_alpha, max_iter=10000)
    lassoModel.fit(x_train, y_train)

    feature_name = data.drop(columns=["SalePrice"]).columns
    coeffs = lassoModel.coef_

    coeffs_df = pd.DataFrame({
        "feature": feature_name,
        "coeffs": coeffs
    })

    coeffs_df['abs_coef'] = coeffs_df['coeffs'].abs()
    coeffs_df = coeffs_df.sort_values(by="abs_coef", ascending=False)

    most_important_features = coeffs_df.iloc[0]
    print(f'Самый важный признак: {most_important_features["feature"]}.'
          f' Коэффициент: {most_important_features["coeffs"]}')

    plt.figure(figsize=(5, 6))
    sns.barplot(x="abs_coef", y="feature", data=coeffs_df.head(5))
    plt.title('Рейтинг признаков по важности')
    plt.show()


if __name__ == "__main__":
    main()
