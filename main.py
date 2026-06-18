import cv2
import mediapipe as mp
import math
import psutil
import time


# =========================================================
# SISTEMA
# =========================================================

cpu = 0
ram = 0
disco = 0

ultimo_update = 0



def actualizar_stats():

    global cpu, ram, disco

    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    disco = psutil.disk_usage("/")



def tick():

    global ultimo_update

    ahora = time.time()

    if ahora - ultimo_update > 1:

        actualizar_stats()
        ultimo_update = ahora



# =========================================================
# UTILIDADES
# =========================================================

def distancia(p1,p2):

    return math.hypot(
        p1.x-p2.x,
        p1.y-p2.y
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

    num_faces=1

)



landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    options
)


print("Modelo cargado")



# =========================================================
# CAMARA
# =========================================================

camara = cv2.VideoCapture(0)



# =========================================================
# CALIBRACION
# =========================================================

calibrando = True

inicio_calibracion = time.time()

muestras = []


base_ceja_izq = 0
base_ceja_der = 0



# =========================================================
# LOOP
# =========================================================

while True:


    ret,frame = camara.read()


    if not ret:
        break



    tick()



    h,w = frame.shape[:2]



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



    if resultado.face_landmarks:


        puntos = resultado.face_landmarks[0]



        # =================================================
        # PUNTOS
        # =================================================

        for p in puntos:


            cv2.circle(

                frame,

                (
                int(p.x*w),
                int(p.y*h)
                ),

                1,

                (0,0,255),

                -1

            )



        # =================================================
        # MEDICION CEJAS
        # =================================================


        ceja_izq = distancia(

            puntos[105],

            puntos[159]

        )


        ceja_der = distancia(

            puntos[334],

            puntos[386]

        )



        altura = distancia(

            puntos[10],

            puntos[152]

        )



        valor_izq = ceja_izq / altura

        valor_der = ceja_der / altura



        # ----------------------------
        # CALIBRACION
        # ----------------------------


        if calibrando:


            muestras.append(

                (
                valor_izq,

                valor_der

                )

            )



            if time.time()-inicio_calibracion > 3:


                base_ceja_izq = sum(
                    x[0] for x in muestras
                ) / len(muestras)



                base_ceja_der = sum(
                    x[1] for x in muestras
                ) / len(muestras)



                calibrando=False


                print("Calibracion completa")



            cv2.putText(

                frame,

                "CALIBRANDO... rostro neutro",

                (40,50),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0,255,255),

                2

            )



        else:


            ceja_izq_control = (

                valor_izq-base_ceja_izq

            ) * 10



            ceja_der_control = (

                valor_der-base_ceja_der

            ) * 10



            ceja_izq_control = max(
                0,
                min(1,ceja_izq_control)
            )


            ceja_der_control = max(
                0,
                min(1,ceja_der_control)
            )



            cv2.putText(

                frame,

                f"Ceja IZQ: {ceja_izq_control:.2f}",

                (20,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0,255,0),

                1

            )


            cv2.putText(

                frame,

                f"Ceja DER: {ceja_der_control:.2f}",

                (20,65),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0,255,0),

                1

            )



        # =================================================
        # BOCA
        # =================================================


        boca = distancia(

            puntos[13],

            puntos[14]

        )


        boca_valor = min(
            1,
            boca / 0.08
        )



        cv2.putText(

            frame,

            f"Boca: {boca_valor:.2f}",

            (20,90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0,255,0),

            1

        )




    cv2.imshow(
        "Face Controller",
        frame
    )


    if cv2.waitKey(1)==27:

        break



camara.release()

cv2.destroyAllWindows()