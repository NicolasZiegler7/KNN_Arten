path = "../Data/iris.csv"
data = pd.read_csv(path, delimiter=',')
print(data.head())

print("Empty columns: ", data.columns[data.isnull().any()])

# Ausgabe der Korrelationen
correlations = data[data.columns].corr(numeric_only=True)
print('All correlations')
print('-' * 30)
correlations_abs_sum = correlations[correlations.columns].abs().sum()
print(correlations_abs_sum)
print('Weakest correlations')
print('-' * 30)
print(correlations_abs_sum.nsmallest(5))

"""Daten vorbereiten."""

# data['sepal.width'] könnte man evtl weglassen, hat geringe Korrelation -> testen
data = data.drop(['sepal.width'], axis = 1)

# Diese Spalte soll vorhergesagt werden
col = data['sepal.length']
data = data.drop(['sepal.length'], axis = 1)

# führe OHE für diese Daten durch
conv_ohe = ['species']
data = pd.get_dummies(data, columns = conv_ohe, dtype=float)

# Erzeuge Objekt
s_scaler = StandardScaler()
# Spalten für StandardScaler
cols_to_s_scale = ['petal.length', 'petal.width']
data[cols_to_s_scale] = s_scaler.fit_transform(data[cols_to_s_scale])

"""KNN aufbauen"""

# Aus den zwei Tabellen vier Tabellen erzeugen
train_data, test_data, train_col, test_col = train_test_split(data,col, test_size=0.2, random_state=42)

# Aufbau KNN
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(data.shape[1],)))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(1))

# Konfiguration des Lernprozesses
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

"""Trainieren"""

# 70 Durchläufe
model.fit(train_data, train_col, epochs=70)

"""Testen"""

test_loss, test_mae = model.evaluate(test_data, test_col)
print('Test mae:', test_mae)