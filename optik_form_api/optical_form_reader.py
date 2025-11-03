# Bu dosyanın adı: optical_form_reader.py

import cv2
import numpy as np
from sklearn.cluster import KMeans


# --- AŞAMA 1: PERSPEKTİF DÜZELTME ---

def order_points(pts):
    """(Yardımcı Fonksiyon) Köşeleri sıralar."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(image_contents: bytes, target_width=1440, target_height=2570):
    """
    1. AŞAMA: Görüntü byte'larını yükler, perspektifi düzeltir ve hedef boyutlara getirir.
    """
    # Gelen byte verisini (contents) OpenCV'nin okuyabileceği bir numpy dizisine çevir
    nparr = np.frombuffer(image_contents, np.uint8)
    # Görüntüyü renkli olarak (IMREAD_COLOR) çöz
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        print("Hata: Gelen byte verisi (contents) resme dönüştürülemedi.")
        return None

    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_contour = None
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 5000:
            doc_contour = approx
            break

    if doc_contour is None:
        print("Hata: Ana kağıt çerçevesi bulunamadı.")
        return None

    rect_points = order_points(doc_contour.reshape(4, 2))
    (tl, tr, br, bl) = rect_points
    widthA = np.linalg.norm(br - bl);
    widthB = np.linalg.norm(tr - tl)
    avg_width = (widthA + widthB) / 2
    heightA = np.linalg.norm(tr - br);
    heightB = np.linalg.norm(tl - bl)
    avg_height = (heightA + heightB) / 2

    if avg_width > avg_height:
        print("Yatay görüntü algılandı. 90 derece döndürülüyor.")
        rect_points = np.roll(rect_points, -1, axis=0)
    else:
        print("Dikey görüntü algılandı. Normal işleniyor.")

    target_points = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect_points, target_points)
    warped = cv2.warpPerspective(orig, matrix, (target_width, target_height))

    print("Aşama 1: Perspektif düzeltme tamamlandı.")
    return warped


# --- AŞAMA 2 & 3: GRID HESAPLAMA VE OKUMA (Debug kodları kaldırıldı) ---

def find_grid_centers_kmeans(circles_cluster, num_rows, num_cols):
    """
    K-Means kümelemesi kullanarak ızgara merkezlerini bulur.
    """
    if not circles_cluster or len(circles_cluster) < max(num_rows, num_cols):
        return None, None, None

    circles_cluster = np.array(circles_cluster)

    try:
        # 1. Y ekseni (Satırlar) için merkezleri bul
        y_coords = circles_cluster[:, 1].reshape(-1, 1)
        kmeans_y = KMeans(n_clusters=num_rows, n_init=10, random_state=0).fit(y_coords)
        row_centers_y = np.sort(kmeans_y.cluster_centers_.flatten())

        # 2. X ekseni (Sütunlar) için merkezleri bul
        x_coords = circles_cluster[:, 0].reshape(-1, 1)
        kmeans_x = KMeans(n_clusters=num_cols, n_init=10, random_state=0).fit(x_coords)
        col_centers_x = np.sort(kmeans_x.cluster_centers_.flatten())

        # 3. Ortalama yarıçapı bul
        avg_radius = int(np.mean(circles_cluster[:, 2]))
        read_radius = int(avg_radius * 0.8)

        return col_centers_x, row_centers_y, read_radius
    except ValueError as e:
        print(f"KMeans Hatası: {e}. Yeterli küme bulunamadı.")
        return None, None, None


def read_student_no_kmeans(thresh, col_centers, row_centers, radius):
    """
    K-Means ile bulunan GERÇEK ızgara merkezlerine göre okur. (Debug kaldırıldı)
    """
    min_pixel_threshold = (np.pi * (radius ** 2)) * 0.3
    student_number = ""

    for col_idx, cx in enumerate(col_centers):
        pixel_counts = []
        for row_idx, cy in enumerate(row_centers):
            cx, cy = int(cx), int(cy)
            r = radius
            y1, y2 = cy - r, cy + r
            x1, x2 = cx - r, cx + r
            y1, y2 = max(0, y1), min(thresh.shape[0], y2)
            x1, x2 = max(0, x1), min(thresh.shape[1], x2)
            roi = thresh[y1:y2, x1:x2]
            filled_pixels = cv2.countNonZero(roi)
            pixel_counts.append(filled_pixels)

        max_filled_pixels = max(pixel_counts)
        if max_filled_pixels > min_pixel_threshold:
            detected_digit = pixel_counts.index(max_filled_pixels)
            student_number += str(detected_digit)
        else:
            student_number += 'X'
    return student_number


def get_dynamic_grid_params(circles_cluster, num_rows, num_cols):
    """
    Cevaplar bölümü için min/max'a dayalı (hızlı) ızgara hesaplayıcı.
    """
    if not circles_cluster or len(circles_cluster) < max(num_rows, num_cols):
        return None
    circles_cluster = np.array(circles_cluster)
    min_x = np.min(circles_cluster[:, 0]);
    max_x = np.max(circles_cluster[:, 0])
    min_y = np.min(circles_cluster[:, 1]);
    max_y = np.max(circles_cluster[:, 1])
    avg_radius = int(np.mean(circles_cluster[:, 2]))
    grid_start_x = min_x;
    grid_start_y = min_y
    step_x = (max_x - min_x) / (num_cols - 1) if num_cols > 1 else 0
    step_y = (max_y - min_y) / (num_rows - 1) if num_rows > 1 else 0
    return {
        "start_x": int(grid_start_x), "start_y": int(grid_start_y),
        "step_x": int(step_x), "step_y": int(step_y),
        "radius": int(avg_radius * 0.8)
    }


def read_answers_relative(thresh, grid_params):
    """
    Cevapları min/max'a dayalı ızgara parametrelerine göre okur. (Debug kaldırıldı)
    """
    p = grid_params
    num_rows = 10;
    num_cols = 5
    min_pixel_threshold = (np.pi * (p['radius'] ** 2)) * 0.3
    answers = {};
    options_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    for row in range(num_rows):
        pixel_counts = []
        for col in range(num_cols):
            cx = int(p['start_x'] + (col * p['step_x']))
            cy = int(p['start_y'] + (row * p['step_y']))
            r = p['radius']
            y1, y2 = cy - r, cy + r;
            x1, x2 = cx - r, cx + r
            y1, y2 = max(0, y1), min(thresh.shape[0], y2)
            x1, x2 = max(0, x1), min(thresh.shape[1], x2)
            roi = thresh[y1:y2, x1:x2];
            filled_pixels = cv2.countNonZero(roi)
            pixel_counts.append(filled_pixels)

        max_filled_pixels = max(pixel_counts)
        if max_filled_pixels > min_pixel_threshold:
            marked_option_index = pixel_counts.index(max_filled_pixels)
            answers[row] = options_map[marked_option_index]
        else:
            answers[row] = 'BOŞ'
    return answers


# --- ANA API FONKSİYONU ---

def formu_analiz_et(image_contents: bytes):
    """
    FastAPI tarafından çağrılan ana fonksiyon.
    Görüntü byte'larını alır, işler ve sonuçları bir dict olarak döner.
    """

    # 1. AŞAMA: Perspektifi Düzelt
    warped_image = perspective_transform(image_contents)
    if warped_image is None:
        # Hata durumunda FastAPI'nin yakalaması için bir Exception fırlat
        raise Exception("Aşama 1 Başarısız: Görüntü düzeltilemedi veya ana kağıt çerçevesi bulunamadı.")

    # 2. AŞAMA: TÜM DAİRELERİ BUL (HoughCircles)
    print("Aşama 2: Daireler (baloncuklar) aranıyor...")
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_warped, (5, 5), 0)

    min_dist = 80;
    min_rad = 35;
    max_rad = 45;
    param1_canny = 100;
    param2_perfection = 25

    circles = cv2.HoughCircles(
        blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=param1_canny, param2=param2_perfection,
        minRadius=min_rad, maxRadius=max_rad
    )

    if circles is None:
        raise Exception("Aşama 2 Başarısız: Görüntüde hiç daire bulunamadı. Lütfen daha net bir fotoğraf çekin.")

    circles = np.uint16(np.around(circles[0, :]))
    print(f"Toplam {len(circles)} adet daire bulundu.")

    # 3. AŞAMA: DAHA AKILLI KÜMELEME (Y Boşluğuna Göre)
    print("Aşama 3: Daireler 'boşluklara' göre kümeleniyor...")
    student_no_circles = []
    answers_left_circles = []
    answers_right_circles = []

    circles_sorted_y = sorted(circles, key=lambda c: c[1])
    y_coords = [c[1] for c in circles_sorted_y]
    y_diffs = np.diff(y_coords)

    if len(y_diffs) == 0:
        raise Exception("Aşama 3 Başarısız: Kümeleme için yeterli daire bulunamadı.")

    largest_gap_index = np.argmax(y_diffs)
    top_cluster = circles_sorted_y[:largest_gap_index + 1]
    bottom_cluster = circles_sorted_y[largest_gap_index + 1:]

    mid_x = warped_image.shape[1] / 2
    for c in bottom_cluster:
        if c[0] < mid_x:
            answers_left_circles.append(c)
        else:
            answers_right_circles.append(c)

    top_cluster_sorted_x = sorted(top_cluster, key=lambda c: c[0])
    x_coords = [c[0] for c in top_cluster_sorted_x]
    x_diffs = np.diff(x_coords)
    largest_x_gap_index = np.argmax(x_diffs)

    if x_diffs[largest_x_gap_index] > 150:
        student_no_circles = top_cluster_sorted_x[largest_x_gap_index + 1:]
    else:
        student_no_circles = top_cluster

    print(f"Öğrenci No Grubu: {len(student_no_circles)} daire (temizlendi)")
    print(f"Cevaplar Sol Grubu: {len(answers_left_circles)} daire")
    print(f"Cevaplar Sağ Grubu: {len(answers_right_circles)} daire")

    # Okuma için eşiklenmiş görüntüyü hazırla
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Dönecek olan JSON yanıtını oluştur
    final_results = {
        "ogrenci_no": "OKUNAMADI",
        "cevaplar": {}
    }

    # 4. AŞAMA: OKUMA
    print("Aşama 4: Izgaralar (grid) hesaplanıyor ve okunuyor...")

    # Öğrenci No Oku
    if len(student_no_circles) > 50:  # 10x10=100
        col_centers, row_centers, radius = find_grid_centers_kmeans(student_no_circles, 10, 10)
        if col_centers is not None and row_centers is not None:
            student_no = read_student_no_kmeans(thresh, col_centers, row_centers, radius)
            final_results['ogrenci_no'] = student_no

    # Cevapları birleştir
    all_answers = {}

    # Sol Cevapları (1-10) Oku
    if len(answers_left_circles) > 25:  # 10x5=50
        params = get_dynamic_grid_params(answers_left_circles, 10, 5)
        if params:
            answers_left = read_answers_relative(thresh, params)
            for i, answer in answers_left.items():
                all_answers[str(i + 1)] = answer  # JSON key'leri string olmalı

    # Sağ Cevapları (11-20) Oku
    if len(answers_right_circles) > 25:  # 10x5=50
        params = get_dynamic_grid_params(answers_right_circles, 10, 5)
        if params:
            answers_right = read_answers_relative(thresh, params)
            for i, answer in answers_right.items():
                all_answers[str(i + 11)] = answer

    final_results['cevaplar'] = all_answers

    print(f"Analiz tamamlandı. Sonuç: {final_results}")

    # Sonuçları FastAPI'ye dictionary olarak döndür
    return final_results