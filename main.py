import cv2
import mediapipe as mp
import math
import psutil
import time


# =========================================================
# MONITOREO DEL SISTEMA
# =========================================================

cpu = 0
ram = 0
disco = 0

ultimo_update = 0


def actualizar_stats():
    global cpu, ram, disco

    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disco = psutil.disk_usage("/").percent


def tick():
    global ultimo_update

    ahora = time.time()

    if ahora - ultimo_update > 1:
        actualizar_stats()
        ultimo_update = ahora



# =========================================================
# UTILIDADES
# =========================================================

def distancia(p1, p2):

    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )



# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=3
)


landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    options
)


print("Modelo cargado")



# =========================================================
# CÁMARA
# =========================================================

camara = cv2.VideoCapture(0)


if not camara.isOpened():

    raise RuntimeError(
        "No se pudo abrir la cámara"
    )



# =========================================================
# LOOP PRINCIPAL
# =========================================================

while True:


    ret, frame = camara.read()


    if not ret:

        break



    tick()


    h, w = frame.shape[:2]



    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )



    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )



    resultado = landmarker.detect_for_video(
        mp_image,
        int(camara.get(cv2.CAP_PROP_POS_MSEC))
    )



    # =====================================================
    # DETECCIÓN DE CARAS
    # =====================================================

    if resultado.face_landmarks:



        for idx, puntos in enumerate(resultado.face_landmarks):


            # -------------------------------------------------
            # LANDMARKS FACIALES
            # -------------------------------------------------

            for punto in puntos:


                px = int(
                    punto.x * w
                )


                py = int(
                    punto.y * h
                )


                cv2.circle(
                    frame,
                    (px, py),
                    1,
                    (0, 0, 255),
                    -1
                )



            # -------------------------------------------------
            # OJOS
            # -------------------------------------------------

            ojo_izq = distancia(
                puntos[386],
                puntos[374]
            )


            ojo_der = distancia(
                puntos[159],
                puntos[145]
            )



            limite = 0.015



            if ojo_izq < limite and ojo_der < limite:

                estado_ojos = "cerrados"


            elif ojo_izq < limite:

                estado_ojos = "izq cerrado"


            elif ojo_der < limite:

                estado_ojos = "der cerrado"


            else:

                estado_ojos = "abiertos"



            # -------------------------------------------------
            # BOCA
            # -------------------------------------------------

            boca = distancia(
                puntos[13],
                puntos[14]
            )



            estado_boca = (

                "abierta"

                if boca > 0.035

                else "cerrada"

            )



            # -------------------------------------------------
            # POSICIÓN HUD
            # -------------------------------------------------

            nariz = puntos[1]


            x = int(
                nariz.x * w
            )


            y = int(
                nariz.y * h
            )



            cv2.putText(
                frame,
                f"Persona {idx}",
                (x + 20, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,255),
                1
            )


            cv2.putText(
                frame,
                f"Ojos: {estado_ojos}",
                (x + 20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0,255,0),
                1
            )


            cv2.putText(
                frame,
                f"Boca: {estado_boca}",
                (x + 20, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0,255,0),
                1
            )



    # =====================================================
    # NO PERSON DETECTED
    # =====================================================

    else:


        overlay = frame.copy()


        x1 = int(w / 2) - 220
        y1 = int(h / 2) - 45


        x2 = int(w / 2) + 220
        y2 = int(h / 2) + 45



        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0,0,255),
            -1
        )



        alpha = 0.35



        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1-alpha,
            0
        )



        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (x1 + 25, int(h/2)+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )



    # =====================================================
    # HUD SISTEMA
    # =====================================================

    cv2.rectangle(
        frame,
        (5,5),
        (170,85),
        (0,0,0),
        -1
    )


    cv2.putText(
        frame,
        f"CPU: {cpu}%",
        (10,25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )


    cv2.putText(
        frame,
        f"RAM: {ram}%",
        (10,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )


    cv2.putText(
        frame,
        f"DISCO: {disco}%",
        (10,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )



    cv2.imshow(
        "Face Tracker",
        frame
    )



    if cv2.waitKey(1) == 27:

        break



camara.release()

cv2.destroyAllWindows()