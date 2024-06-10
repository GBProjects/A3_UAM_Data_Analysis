import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Carregar o arquivo CSV
file_path = 'C:/Users/User/Desktop/Aulas vscode/A3/Walmart_Sales.csv'
data = pd.read_csv(file_path)

# Converter a coluna Date para datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Verificar valores ausentes
print(data.isnull().sum())

# Codificar Holiday_Flag
data['Holiday_Flag'] = data['Holiday_Flag'].astype(int)

# Normalizar variáveis numéricas
numeric_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Verificar a distribuição de Weekly_Sales
sns.histplot(data['Weekly_Sales'], bins=50, kde=True)
plt.show()

# Calcular a matriz de correlação
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Detectar e remover outliers em Weekly_Sales usando IQR diretamente
Q1 = data['Weekly_Sales'].quantile(0.25)
Q3 = data['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Weekly_Sales'] >= lower_bound) & (data['Weekly_Sales'] <= upper_bound)]

# Seleção de variáveis
X = data.drop(['Weekly_Sales', 'Date'], axis=1)  # Removendo a coluna 'Date'
y = data['Weekly_Sales']

# Verificar tipos de dados
print(X.dtypes)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar modelo de regressão com GridSearchCV
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(grid_search.best_params_)

# Avaliação do modelo no conjunto de treino
y_train_pred = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f'Train Mean Squared Error: {train_mse}')
print(f'Train R2 Score: {train_r2}')

# Avaliação do modelo no conjunto de teste
y_test_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f'Test Mean Squared Error: {test_mse}')
print(f'Test R2 Score: {test_r2}')

# Visualização dos resultados no conjunto de teste
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales')
plt.plot(y_test_pred, label='Predicted Sales')
plt.xlabel('Weeks')
plt.ylabel('Weekly Sales')
plt.title('Actual vs Predicted Weekly Sales')
plt.legend()
plt.show()

# Visualização adicional: Gráfico de dispersão com linha de tendência
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
