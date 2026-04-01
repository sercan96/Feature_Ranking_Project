""""" Info: Models dosyası, model tanımlamalarını ve eğitimi için kullanılır."""

from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_baseline_model():
    """
    Baseline olarak Logistic Regression modeli oluşturur.
    """
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    return model

#CNN modeli oluşturma fonksiyonu
def build_feature_ranking_cnn(input_shape):
    """
    Basit ve açıklanabilir CNN modeli.
    İlk Conv1D katmanı kernel_size=1 olduğu için feature ranking daha yorumlanabilir hale gelir.
    """
    inputs = Input(shape=input_shape, name="input_layer")

    # Feature ranking için en önemli katman:
    # İlk conv katmanı -> her feature için öğrenilen ağırlıkları buradan çıkaracağız
    x = Conv1D(
        filters=32,
        kernel_size=input_shape[0],  # Her seferinde tüm feature’lara bakacak şekilde kernel_size=feature sayısı
        padding="same",
        name="feature_conv"
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Birkaç basit conv katmanı daha
    """
    64 filtre → 64 farklı pattern öğrenilecek (F1, F2, F3... birlikte nasıl etkili  1.(F1-F2-F3) 2. (F1-F2-F4) gibi)
    kernel_size=3 → her seferinde 3 feature’a birlikte bakıyor. F1-F2-F3 gibi.
    padding="same" → output boyutu input ile aynı kalır
    Activation("relu")(x) => negatifleri sıfırlar,pozitifleri bırakır.Gürültüyü azaltır. [-2, -1, 0, 3, 5] → [0, 0, 0, 3, 5]
    1️1.Pattern çıkarır (Conv1D)
    2️2. Stabil hale getirir (BatchNorm)
    3️3. Önemli sinyali bırakır (ReLU)
    """
    x = Conv1D(filters=64, kernel_size=input_shape[0], padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)   

    x = Conv1D(filters=64, kernel_size=input_shape[0], padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="feature_ranking_cnn")
    return model

"""
AutoEncoder Mimarisi: 
Input → Encoder → Bottleneck → Decoder → Output
"""
def build_autoencoder(input_dim=30, encoding_dim=8): #boyutu 30’dan 8’e düşürür
    """
    Tablosal veri için dense autoencoder modeli oluşturur.
    Geriye:
    - autoencoder modeli
    - encoder modeli
    döndürür.
    """
    input_layer = Input(shape=(input_dim,), name="input_layer")

    # Encoder
    encoded = Dense(16, activation="relu", name="encoder_dense_1")(input_layer) #Veriyi 30’dan 16’ya indiriyor.
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(encoded)

    # Decoder
    decoded = Dense(16, activation="relu", name="decoder_dense_1")(bottleneck)
    output_layer = Dense(input_dim, activation="linear", name="output_layer")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    encoder = Model(inputs=input_layer, outputs=bottleneck, name="encoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return autoencoder, encoder


__all__ = ["build_baseline_model", "build_feature_ranking_cnn", "build_autoencoder"]