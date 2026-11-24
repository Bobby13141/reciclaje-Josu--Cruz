import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Clasificador de reciclaje")

MODEL_DIR = Path("modelo_reciclaje_mobilenet")
MODEL_PATH_H5 = MODEL_DIR / "waste_mobilenet.h5"
MODEL_PATH_KERAS = MODEL_DIR / "waste_mobilenet.keras"
CLASSES_PATH = MODEL_DIR / "class_names.json"
IMG_SIZE = (224, 224)

LABELS_ES = {
    "cardboard": "Cartón",
    "glass": "Vidrio",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "trash": "Basura",
}


def load_class_names() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error(f"No se encontró el archivo de clases: {CLASSES_PATH}")
        st.stop()
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mobilenet_model(num_classes: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,
    )
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@st.cache_resource
def load_model(num_classes: int):
    if not MODEL_PATH_H5.exists() and not MODEL_PATH_KERAS.exists():
        st.error("No se encontró el modelo en 'modelo_reciclaje_mobilenet'.")
        st.stop()

    if MODEL_PATH_H5.exists():
        try:
            model = build_mobilenet_model(num_classes)
            model.load_weights(MODEL_PATH_H5)
            return model
        except Exception:
            pass

    try:
        path = MODEL_PATH_KERAS if MODEL_PATH_KERAS.exists() else MODEL_PATH_H5
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_top3(model, class_names: list[str], image: Image.Image):
    x = prepare_image(image)
    preds = model.predict(x, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    results = []
    for idx in top3_idx:
        raw_label = class_names[idx]
        show_label = LABELS_ES.get(raw_label, raw_label)
        results.append((raw_label, show_label, float(preds[idx]) * 100))
    return results


st.title("♻️ Clasificador de residuos")
st.subheader("Computación en la Nube - UTH 2026 - Josué Miguel Cruz Juánez")
st.write("Toma una fotografía con la cámara o sube una imagen para identificar el tipo de residuo.")
st.caption("Clases: cartón, vidrio, metal, papel, plástico y basura general.")

class_names = load_class_names()
model = load_model(len(class_names))

metodo = st.radio("Elige cómo ingresar la imagen:", ["Subir archivo", "Tomar fotografía con cámara"], horizontal=True)

imagen_fuente = None
if metodo == "Subir archivo":
    imagen_fuente = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
else:
    imagen_fuente = st.camera_input("Captura una fotografía del residuo")

if imagen_fuente is not None:
    image = Image.open(imagen_fuente)
    st.image(image, caption="Imagen analizada", use_container_width=True)

    top3 = predict_top3(model, class_names, image)
    principal = top3[0]

    st.subheader("Predicción principal")
    st.success(f"{principal[1]}: {principal[2]:.2f}%")

    st.subheader("Top 3")
    for _, label_es, score in top3:
        st.write(f"- {label_es}: {score:.2f}%")

    st.subheader("Detalle por clase")
    x = prepare_image(image)
    preds = model.predict(x, verbose=0)[0]
    for i, raw_label in enumerate(class_names):
        label_es = LABELS_ES.get(raw_label, raw_label)
        st.progress(float(preds[i]), text=f"{label_es}: {preds[i]*100:.2f}%")
else:
    st.info("Carga o captura una imagen para comenzar.")
