# Alle object-Spalten automatisch umwandeln
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes


cb_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.fit(train_data, train_col, epochs=100, validation_data=(test_data, test_col),callbacks=[cb_early])
