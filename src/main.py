import cv2
import numpy as np
import math
import webbrowser


def deseneazaLinii(img, lines, culoare):
    if lines is not None:
        for v_line in lines:
            # compute the lines
            rho = v_line[0][0]
            theta = v_line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho

            aux = max(img.shape[0], img.shape[1])  # maximul dintre latimea si inaltimea pozei
            pct1 = (int(x0 + aux * (-b)), int(y0 + aux * (a)))
            pct2 = (int(x0 - aux * (-b)), int(y0 - aux * (a)))

            cv2.line(img, pct1, pct2, culoare, 1)


def cautaLinii(img, min_theta, max_theta, nr_min_linii=9, nr_max_linii=25):
    if nr_min_linii < 9:
        nr_min_linii = 9
    if nr_max_linii > 25:
        nr_max_linii = 25
    if nr_min_linii > nr_max_linii:
        nr_min_linii = 9

    aparitii = np.zeros(1000)  # mare, ca sa fie suficient
    threshold_adaptiv = 150
    aparitii[threshold_adaptiv] += 1
    lines = cv2.HoughLines(img, 1, (np.pi / 180), threshold_adaptiv, None, 0, 0, (np.pi / 180) * min_theta,
                           (np.pi / 180) * max_theta)

    while len(lines) < nr_min_linii or len(lines) > nr_max_linii:
        if len(lines) < nr_min_linii:
            threshold_adaptiv -= 1  # mai indulgenti la linii
        if len(lines) > nr_max_linii:
            threshold_adaptiv += 1  # mai multe linii sa fie bine sa nu fie rau
        aparitii[threshold_adaptiv] += 1

        # daca threshold se repeta, luam varianta cu cele mai multe linii, mai bine mai multe linii decat mai putine
        if aparitii[threshold_adaptiv] > 1:
            while len(lines) < nr_min_linii:
                threshold_adaptiv += 1
                lines = cv2.HoughLines(img, 1, (np.pi / 180), threshold_adaptiv, None, 0, 0, (np.pi / 180) * min_theta,
                                       (np.pi / 180) * max_theta)

        lines = cv2.HoughLines(img, 1, (np.pi / 180), threshold_adaptiv, None, 0, 0, (np.pi / 180) * min_theta,
                               (np.pi / 180) * max_theta)

    return lines


def intersecteazaLinii(verticale, orizontale):
    puncteLinii = []
    for linie1 in orizontale:
        puncte_linie = []
        for linie2 in verticale:
            rho1, theta1 = linie1[0]
            rho2, theta2 = linie2[0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            puncte_linie.append((x0, y0))

        puncte_linie.sort(key=lambda x: x[0])  # sortam sa ne ajute la eliminat
        puncteLinii.append(puncte_linie)

    puncteLinii.sort(key=lambda x: x[0][1])

    puncteColoane = []
    for linie1 in verticale:
        puncte_coloana = []
        for linie2 in orizontale:
            rho1, theta1 = linie1[0]
            rho2, theta2 = linie2[0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            puncte_coloana.append((x0, y0))

        puncte_coloana.sort(key=lambda x: x[1])  # sortam sa ne ajute la eliminat
        puncteColoane.append(puncte_coloana)

    puncteColoane.sort(key=lambda x: x[0][1])

    return puncteLinii, puncteColoane


def calculeazaRating(puncteLista):
    ratingPuncte = []  # index pct pe linie/coloana sa, rating
    distante = []
    for i in range(1, len(puncteLista)):
        x0, y0 = puncteLista[i - 1]
        x1, y1 = puncteLista[i]
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        distante.append(dist)

    distante.sort()
    medLinie = distante[int(len(distante) / 2)]

    for i in range(0, len(puncteLista)):
        x_act, y_act = puncteLista[i]
        if i == 0 and len(puncteLista) > 2:
            x_urm, y_urm = puncteLista[i + 1]
            x_ant, y_ant = x_urm, y_urm
        elif i == len(puncteLista) - 1 and len(puncteLista) > 2:
            x_ant, y_ant = puncteLista[i - 1]
            x_urm, y_urm = x_ant, y_ant
        else:
            x_urm, y_urm = puncteLista[i + 1]
            x_ant, y_ant = puncteLista[i - 1]

        dist_ant = math.sqrt((x_act - x_ant) ** 2 + (y_act - y_ant) ** 2)
        dist_urm = math.sqrt((x_act - x_urm) ** 2 + (y_act - y_urm) ** 2)
        rating = max(abs(medLinie - dist_urm),
                     abs(medLinie - dist_ant))  # practic cat de distantat e pct fata de vecini, raportat la medie

        ratingPuncte.append((i, rating))

    ratingPuncte.sort(key=lambda x: x[1])  # sortam in functie de rating, primul e cel mai probabil sa fie gresit
    ratingPuncte.reverse()

    return ratingPuncte


def filtreazaPuncte(puncte, maxPuncte=9):
    puncteLinii, puncteColoane = puncte
    puncteCorectate = []

    for i in range(0, len(puncteLinii)):
        while len(puncteLinii[i]) > maxPuncte:
            ratingPuncte = calculeazaRating(puncteLinii[i])
            j = ratingPuncte[0][0]

            for linie in puncteLinii:
                linie.pop(j)

            puncteColoane.pop(j)

    for j in range(0, len(puncteColoane)):
        while len(puncteColoane[j]) > maxPuncte:
            ratingPuncte = calculeazaRating(puncteColoane[j])
            i = ratingPuncte[0][0]
            for coloana in puncteColoane:
                coloana.pop(i)

            puncteLinii.pop(i)

    return puncteCorectate


def deseneazaCercuri(img, matriceDePuncte=None, vectorDePuncte=None):
    if matriceDePuncte is not None:
        for linie_cu_puncte in matriceDePuncte:
            for centru in linie_cu_puncte:
                cv2.circle(img, centru, 3, (255, 0, 255), 2)
    elif vectorDePuncte is not None:
        for centru in vectorDePuncte:
            cv2.circle(img, centru, 3, (255, 0, 255), 2)


def caculeazaDreptunghi(dreptunghi):
    cel_mai_sus = min(dreptunghi[0][1], dreptunghi[1][1])
    cel_mai_jos = max(dreptunghi[2][1], dreptunghi[3][1])
    cel_mai_stang = min(dreptunghi[0][0], dreptunghi[3][0])
    cel_mai_drept = max(dreptunghi[1][0], dreptunghi[1][0])

    height = cel_mai_jos - cel_mai_sus
    width = cel_mai_drept - cel_mai_stang

    new_dreptunghi = np.float32([[0, 0], [width, 0],
                                 [0, height], [width, height]])

    return new_dreptunghi


def corecteazaPerspectiva(img):
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gs, (3, 3))
    img_canny = cv2.Canny(img_blur, 50, 100, 3)

    # cautam liniile
    vertical_lines = cautaLinii(img_canny, -10, 10, 15)
    horizontal_lines = cautaLinii(img_canny, 75, 105, 15)

    # aflam intersectiile lor, alegem doar punctele de pe tabla
    # si ne bazam pe faptul ca colturile extreme ar trebui
    # sa formeze un patrat
    corners = intersecteazaLinii(vertical_lines, horizontal_lines)
    filtreazaPuncte(corners, 10)

    colturi_pe_linii = corners[0]
    m = len(colturi_pe_linii)
    n = len(colturi_pe_linii[0])
    ss = colturi_pe_linii[0][0]
    sj = colturi_pe_linii[m - 1][0]
    ds = colturi_pe_linii[0][n - 1]
    dj = colturi_pe_linii[m - 1][n - 1]

    # marit putin dreptunghiul, sa nu pierdem liniile
    offset = 2

    ss = ((ss[0] - offset), (ss[1] - offset))
    sj = ((sj[0] - offset), (sj[1] + offset))
    ds = ((ds[0] + offset), (ds[1] - offset))
    dj = ((dj[0] + offset), (dj[1] + offset))

    colturi_dreptunghi_vechi = np.float32([ss, ds, sj, dj])
    colturi_dreptunghi_nou = np.float32([[0, 0], [720, 0],
                                         [0, 720], [720, 720]])

    matrix = cv2.getPerspectiveTransform(colturi_dreptunghi_vechi, colturi_dreptunghi_nou)
    correctedPerspective = cv2.warpPerspective(img_orig, matrix, (720, 720))

    return correctedPerspective


def colturileTablei(img):
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gs, (3, 3))
    img_canny = cv2.Canny(img_blur, 50, 100, 3)

    # cautam liniile
    vertical_lines = cautaLinii(img_canny, -10, 10, 9, 14)
    horizontal_lines = cautaLinii(img_canny, 75, 105, 9, 14)

    corners = intersecteazaLinii(vertical_lines, horizontal_lines)
    filtreazaPuncte(corners, 9)

    colturi_pe_linii = corners[0]
    return colturi_pe_linii


def imparteTablaInPatrate(img, colturi):
    patrate = []
    for i in range(0, len(colturi) - 1):
        for j in range(0, len(colturi[i]) - 1):
            x1, y1 = colturi[i][j]
            x2, y2 = colturi[i + 1][j + 1]
            patratel = img[y1:y2, x1:x2]
            patrate.append(patratel)

    return patrate


def verificaPiesa(img, path, threshold):
    template = cv2.imread(path)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    rez = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(rez >= threshold)

    return loc


def analizeazaPiese(img, colturi):
    img = img.copy()
    pieces = [['pieces/white_pawn.png', 'P', 0.27],
              ['pieces/white_knight.png', 'N', 0.32],
              ['pieces/white_bishop.png', 'B', 0.36],
              ['pieces/white_rook.png', 'R', 0.38],
              ['pieces/white_queen.png', 'Q', 0.25],
              ['pieces/white_king.png', 'K', 0.35],
              # ['pieces/black_pawn.png', 'p', 0.7],
              ['pieces/black_knight.png', 'n', 0.20],
              ['pieces/black_bishop.png', 'b', 0.15],
              ['pieces/black_rook.png', 'r', 0.19],
              ['pieces/black_queen.png', 'q', 0.3],
              ['pieces/black_king.png', 'k', 0.19]]

    fen_code = ''

    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i in range(0, len(colturi) - 1):
        line = []
        for j in range(0, len(colturi[i]) - 1):
            x1, y1 = colturi[i][j]
            x2, y2 = colturi[i + 1][j + 1]
            patratel = img_gs[y1:y2, x1:x2]

            probabilitati = []

            for piesa in pieces:
                path, code, threshold = piesa
                loc = verificaPiesa(patratel, path, threshold)
                nr = 0
                for pt in zip(*loc[::-1]):
                    pct1 = (pt[0] + x1, pt[1] + y1)
                    pct2 = (pt[0] + 64 + x1, pt[1] + 64 + y1)
                    # if code == 'bk':
                    #    cv2.rectangle(img, pct1, pct2, (255, 255, 0), 1)
                    nr += 1

                probabilitati.append((code, nr))

            probabilitati.sort(reverse=True, key=lambda x: x[1])

            if probabilitati[0][1] < 10:
                guess = '_'
            else:
                guess = probabilitati[0][0]

            line.append(guess)

        count = 0
        line_code = ''
        for char in line:
            if char != '_':
                if count > 0:
                    line_code += str(count)
                    count = 0
                line_code += char
            else:
                count += 1
        if count > 0:
            line_code += str(count)
            count = 0

        fen_code += line_code
        fen_code += '/'
        # print(line_code)

    fen_code = fen_code[:-1]
    print(fen_code)
    cv2.imshow('', img)

    return fen_code


if __name__ == "__main__":
    # img_orig = cv2.imread("lichess_prtsc.png")
    print("Demo files: poza1.jpg, poza2.jpg, poza3.jpg")
    path_to_img = input('image path/name: ')
    img_orig = cv2.imread(path_to_img)

    img_persp = corecteazaPerspectiva(img_orig)

    colturi = colturileTablei(img_persp)
    deseneazaCercuri(dummy_img, colturi)
    cv2.imshow('', dummy_img)

    fen = analizeazaPiese(img_persp, colturi)

    link = 'https://lichess.org/editor/' + fen
    print("Opening in browser...")
    webbrowser.open(link)

    cv2.waitKey(0)
