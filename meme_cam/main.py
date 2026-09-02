import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque, Counter

mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)

def d(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)

def d2(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)

def esc(lm):
    return d(lm[152], lm[10]) + 1e-6

def px(pt, W, H):
    return (int(pt.x * W), int(pt.y * H))

def dedos_estado(lm, izq=False):
    tip = [8, 12, 16, 20]
    mid_j = [6, 10, 14, 18]
    out = [1 if (lm[4].x > lm[3].x if izq else lm[4].x < lm[3].x) else 0]

    for t, m in zip(tip, mid_j):
        out.append(1 if lm[t].y < lm[m].y else 0)

    return out

class Cal:
    N = 45

    def __init__(self):
        self.buf = {k: [] for k in ['ci', 'cd', 'cen', 'lap', 'llb', 'bi_y', 'bd_y', 'gap']}
        self.done = False
        self.thr = dict(
            ci=0.180, cd=0.180, cen_lo=0.185,
            lap=0.055, llb=0.145,
            bi_y_lo=0.30, bd_y_lo=0.30,
            gap_lo=0.10
        )

    def feed(self, lm):
        if self.done:
            return

        e = esc(lm)
        self.buf['ci'].append(d(lm[52], lm[159]) / e)
        self.buf['cd'].append(d(lm[282], lm[386]) / e)
        self.buf['cen'].append(d(lm[55], lm[285]) / e)
        self.buf['lap'].append(d(lm[13], lm[14]) / e)
        self.buf['llb'].append(d(lm[17], lm[152]) / e)
        self.buf['bi_y'].append(lm[55].y - lm[9].y)
        self.buf['bd_y'].append(lm[285].y - lm[9].y)
        self.buf['gap'].append(abs(lm[55].x - lm[285].x))

        if len(self.buf['ci']) >= self.N:
            self._calc()

    def _calc(self):
        m = lambda k: float(np.median(self.buf[k]))
        s = lambda k: float(np.std(self.buf[k]))
        mg_c = lambda k: max(1.5 * s(k), 0.015)
        mg_b = lambda k, mn: max(3 * s(k), mn)

        self.thr['ci'] = m('ci') + mg_c('ci')
        self.thr['cd'] = m('cd') + mg_c('cd')
        self.thr['cen_lo'] = m('cen') - mg_c('cen')
        self.thr['lap'] = m('lap') + mg_b('lap', 0.032)
        self.thr['llb'] = m('llb') - mg_b('llb', 0.018)
        self.thr['bi_y_lo'] = m('bi_y') + mg_c('bi_y')
        self.thr['bd_y_lo'] = m('bd_y') + mg_c('bd_y')
        self.thr['gap_lo'] = m('gap') - mg_c('gap')
        self.done = True

    @property
    def progress(self):
        return min(len(self.buf['ci']) / self.N, 1.0)


def det_lengua(lm, cal):
    e = esc(lm)
    boca_abierta = d(lm[13], lm[14]) / e > cal.thr['lap']
    lengua_baja = d(lm[17], lm[152]) / e < cal.thr['llb']
    punta_fuera = lm[17].y > lm[14].y + 0.012
    return boca_abierta and lengua_baja and punta_fuera


def det_ceja(lm, cal):
    e = esc(lm)
    ci = d(lm[52], lm[159]) / e
    cd = d(lm[282], lm[386]) / e
    cen = d(lm[55], lm[285]) / e
    bi_y = lm[55].y - lm[9].y
    bd_y = lm[285].y - lm[9].y
    gap = abs(lm[55].x - lm[285].x)

    return (
        ci > cal.thr['ci'] or
        cd > cal.thr['cd'] or
        cen < cal.thr['cen_lo'] or
        bi_y > cal.thr['bi_y_lo'] or
        bd_y > cal.thr['bd_y_lo'] or
        gap < cal.thr['gap_lo']
    )


def det_cristiano(manos, lm_cara):
    boca = lm_cara[13]
    return any(
        d(lm[8], boca) < 0.09 or d(lm[12], boca) < 0.09
        for _, lm in manos
    )


def det_rata(ded):
    return ded == [0, 1, 1, 0, 0]


def dedo_extendido(lm, tip, pip):
    return d2(lm[tip], lm[0]) > d2(lm[pip], lm[0]) * 1.18


def indice_solo(lm):
    return (
        dedo_extendido(lm, 8, 6) and
        not dedo_extendido(lm, 12, 10) and
        not dedo_extendido(lm, 16, 14) and
        not dedo_extendido(lm, 20, 18)
    )


def det_uwu(manos):
    # 👉👈
    if len(manos) != 2 or not all(indice_solo(lm) for _, lm in manos):
        return False

    _, a = manos[0]
    _, b = manos[1]

    dir_a = a[8].x - a[5].x
    dir_b = b[8].x - b[5].x

    return (
        dir_a * dir_b < -0.001 and
        0.035 < abs(a[8].x - b[8].x) < 0.38 and
        abs(a[8].y - b[8].y) < 0.20
    )


def det_who_me(manos, lm_cara):
    # Index finger pointing at your chest.
    centro_x = lm_cara[9].x
    ancho_cara = abs(lm_cara[454].x - lm_cara[234].x)

    for _, lm in manos:
        if indice_solo(lm) and lm[8].y > lm_cara[152].y - 0.03:
            if abs(lm[8].x - centro_x) < ancho_cara * 0.85:
                return True

    return False


def det_dimag(manos, lm_cara):
    # Index finger near forehead or temple.
    alto_cara = d2(lm_cara[10], lm_cara[152])
    objetivos = [lm_cara[10], lm_cara[109], lm_cara[338]]

    for _, lm in manos:
        if indice_solo(lm) and lm[8].y < lm_cara[9].y + alto_cara * 0.25:
            if min(d2(lm[8], punto) for punto in objetivos) < alto_cara * 0.62:
                return True

    return False


def det_gato_dedo_medio(manos):
    # Middle finger up.
    for _, lm in manos:
        if (
            dedo_extendido(lm, 12, 10) and
            not dedo_extendido(lm, 8, 6) and
            not dedo_extendido(lm, 16, 14) and
            not dedo_extendido(lm, 20, 18)
        ):
            return True

    return False


def det_monkey(manos, lm_cara):
    if len(manos) != 2:
        return False

    nariz_y = lm_cara[1].y
    return all(lm[9].y < nariz_y for _, lm in manos)


def det_cara(manos):
    if len(manos) != 2:
        return False

    for ded, lm in manos:
        if ded[1:] != [1, 1, 1, 1] or lm[0].y < 0.50:
            return False

    return abs(manos[0][1][0].x - manos[1][1][0].x) >= 0.20


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_hand_minimal(frame, lm, W, H, ded):
    COL = (140, 200, 140)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px(lm[a], W, H), px(lm[b], W, H), COL, 1, cv2.LINE_AA)

    for i in range(21):
        cv2.circle(frame, px(lm[i], W, H), 2, COL, -1, cv2.LINE_AA)

    for i, tip in enumerate([4, 8, 12, 16, 20]):
        if ded[i]:
            cv2.circle(frame, px(lm[tip], W, H), 3, (80, 240, 80), -1, cv2.LINE_AA)


def hud(frame, img_actual, manos_info, W, H):
    nombre = img_actual if img_actual else "neutral"
    col = (80, 220, 80) if img_actual else (160, 160, 160)

    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(W - 8, 14 + len(nombre) * 14 + 20), 36), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, nombre, (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA
    )

    for i, (lado, ded) in enumerate(manos_info):
        cv2.putText(
            frame, f"{lado}: {ded}", (14, 58 + 24 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA
        )


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    ret, frame0 = cap.read()
    if not ret:
        print("Error: no se pudo leer frame inicial")
        cap.release()
        return

    frame0 = cv2.flip(frame0, 1)
    H, W = frame0.shape[:2]

    cv2.namedWindow("Meme Cam", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Meme Cam", frame0)
    cv2.waitKey(1)

    cal = Cal()
    buf = deque(maxlen=10)
    img_actual = None
    MINVOTOS = 6
    meme_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr = face_mesh.process(rgb)
        hr = hands_det.process(rgb)

        det = None
        lm_cara = None
        manos = []
        manos_info = []

        if not cal.done:
            pct = cal.progress
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

            cy = H // 2
            cv2.putText(
                frame, "Mira al frente cara neutral",
                (W // 2 - 200, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2, cv2.LINE_AA
            )

            bx1, bx2 = W // 2 - 140, W // 2 + 140
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (40, 40, 40), -1)
            cv2.rectangle(
                frame, (bx1, cy + 10),
                (bx1 + int(280 * pct), cy + 28),
                (80, 220, 80), -1
            )
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (120, 120, 120), 1)

            cv2.putText(
                frame, f"{int(pct * 100)}%",
                (W // 2 - 18, cy + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (160, 160, 160), 1, cv2.LINE_AA
            )

            if fr.multi_face_landmarks:
                cal.feed(fr.multi_face_landmarks[0].landmark)

            cv2.imshow("Meme Cam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            continue

        if fr.multi_face_landmarks:
            lm_cara = fr.multi_face_landmarks[0].landmark

        if hr.multi_hand_landmarks:
            for i, hl in enumerate(hr.multi_hand_landmarks):
                lm = hl.landmark
                izq = hr.multi_handedness[i].classification[0].label == "Left"
                ded = dedos_estado(lm, izq)

                draw_hand_minimal(frame, lm, W, H, ded)
                manos.append((ded, lm))
                manos_info.append(("I" if izq else "D", ded))

        if lm_cara and len(manos) == 2 and det_monkey(manos, lm_cara):
            det = "monkey.jpg"
        elif len(manos) == 2 and det_uwu(manos):
            det = "uwu.jpg"
        elif len(manos) == 2 and det_cara(manos):
            det = "cara.jpeg"
        elif lm_cara and manos and det_dimag(manos, lm_cara):
            det = "dimag.jpg"
        elif lm_cara and manos and det_cristiano(manos, lm_cara):
            det = "cristiano.png"
        elif lm_cara and manos and det_who_me(manos, lm_cara):
            det = "who_me.jpg"
        elif manos and det_gato_dedo_medio(manos):
            det = "cat.jpg"
        elif lm_cara and det_lengua(lm_cara, cal):
            det = "gato1.png"
        elif lm_cara and det_ceja(lm_cara, cal):
            det = "perro.jpeg"
        elif len(manos) == 1:
            ded_m, lm_m = manos[0]
            if det_rata(ded_m):
                det = "rata.jpeg"

        buf.append(det)
        conteo = Counter(buf)
        top, votos = conteo.most_common(1)[0]

        if votos >= MINVOTOS:
            img_actual = top

        hud(frame, img_actual, manos_info, W, H)

        display = frame.copy()

        if img_actual:
            if img_actual not in meme_cache:
                meme_cache[img_actual] = cv2.imread(img_actual)

            meme = meme_cache[img_actual]

            if meme is not None and meme.size > 0:
                sw, sh = W // 3, H // 3
                meme_small = cv2.resize(meme, (sw, sh))

                x1, x2 = W - sw - 10, W - 10
                y1, y2 = H - sh - 10, H - 10

                cv2.rectangle(
                    display, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                    (0, 0, 0), -1
                )
                display[y1:y2, x1:x2] = meme_small

            else:
                cv2.putText(
                    display, f"Falta: {img_actual}", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 80, 220), 2
                )

        cv2.imshow("Meme Cam", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()