from PIL import Image
import numpy as np
from numpy import asarray
import math
import matplotlib.pyplot as plt
import matplotlib.image
from skimage.segmentation import watershed
from pprint import pprint
import cv2
import pickle
import imutils
import random
from os.path import exists



def initialization_k_means(data, num_clusters):
    clusters = []
    for i in range(num_clusters):
        x = random.randint(0, data.shape[0])
        y = random.randint(0, data.shape[1])
        clusters.append((x,y))
    return clusters


def cluster_assignment(data, clusters):
    classif = []
    classification = {}
    x = data.shape[0]
    y = data.shape[1]
    for i in range(x):
        for j in range(y):
            cluster_dist = []
            for cluster_index, cluster_coords in enumerate(clusters):
                vpr = data[i][j][0]
                vpg = data[i][j][1]
                vpb = data[i][j][2]

                c_x = cluster_coords[0]
                c_y = cluster_coords[1]
                vcr = data[c_x][c_y][0]
                vcg = data[c_x][c_y][1]
                vcb = data[c_x][c_y][2]
                dist = math.sqrt(((vpr - vcr)**2) + ((vpg - vcg)**2) + ((vpb-vcb)**2))

                cluster_dist.append([cluster_index, dist])
            chosen_cluster = min(cluster_dist, key=lambda dist: dist[1])
            classif.append([chosen_cluster[0], i, j])

    for i in range(len(clusters)):
        val = [(x[1], x[2]) for x in classif if x[0] == i]
        if len(val) == 0:
            continue
        classification[i] = val
    return classification


def cluster_centroid(classification, data):
    clusters_rgb = []
    for v in classification.values():
        pixels = [data[x][y] for x, y in v]

        r = np.average([p[0] for p in pixels])
        g = np.average([p[1] for p in pixels])
        b = np.average([p[2] for p in pixels])

        clusters_rgb.append([r, g, b])
    clusters = []
    for i in range(len(clusters_rgb)):
        central_pixel = clusters_rgb[i]
        cluster = ()
        dist_aux = 10000
        x = data.shape[0]
        y = data.shape[1]
        for i in range(x):
            for j in range(y):
                pixel = data[i][j]
                pr = pixel[0] 
                pg = pixel[1]
                pb = pixel[2]  

                cr = central_pixel[0] 
                cg = central_pixel[1]
                cb = central_pixel[2]  
                dist = math.sqrt(((pr - cr)**2) + ((pg-cg)**2) + ((pb-cb)**2))
                if dist < dist_aux:
                    cluster = (i, j)
                    dist_aux = dist
        clusters.append(cluster)
    return clusters
    


def clusterization(img):
    if not exists('classification.pkl'):
        data = np.asarray(img, dtype = 'int64')

        num_clusters = 4
        clusters_init = initialization_k_means(data, num_clusters)

        classification = cluster_assignment(data, clusters_init)
        
        for i in range(10):
            new_clusters = cluster_centroid(classification, data)
            classification = cluster_assignment(data, new_clusters)

        #Guarda el dataset en pickle
        clusters_coords = open ('classification.pkl','wb')
        pickle.dump(classification, clusters_coords)
        clusters_coords.close()
        clf = classification
    else:
        f = open ('classification.pkl','rb')
        clf = pickle.load(f)
        f.close()
    return clf
    

def histogram(image):
    histogram = np.zeros((256), dtype='int')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            aux = int(image[i, j])
            histogram[aux] += 1

    test = 0
    for i in range(256):
        test += histogram[i]
        
    return histogram


def otsu(histogram):
    t2 = 0
    x = 10000.

    total_pixels = 0
    for i in range(256):
        total_pixels += histogram[i]

    for t1 in range(256):
        sigma_w = 0

        # BACK
        Wb = 0
        sum_h_k_b = 0
        Mu_b = 0
        Mu_b_num = 0
        sigma_b = 0
        sigma_b_num = 0

        for i in range(t1):
            Mu_b_num += i * histogram[i]
            sum_h_k_b += histogram[i]
        
        Wb = sum_h_k_b / total_pixels
        try:
            Mu_b = Mu_b_num / sum_h_k_b
        except ZeroDivisionError:
            Mu_b = 0
        

        for i in range(t1):
            sigma_b_num += pow((i - Mu_b), 2) * histogram[i]
        
        try:
            sigma_b = sigma_b_num / sum_h_k_b
        except ZeroDivisionError:
            sigma_b = 0
        

        # FRONT
        Wf = 0
        sum_h_k_f = 0
        Mu_f = 0
        Mu_f_num = 0
        sigma_f = 0
        sigma_f_num = 0

        for i in range(t1, 256):
            Mu_f_num += i * histogram[i]
            sum_h_k_f += histogram[i]
        
        Wf = sum_h_k_f / total_pixels

        try:
            Mu_f = Mu_f_num / sum_h_k_f
        except ZeroDivisionError:
            Mu_f = 0
        

        for i in range(t1, 256):
            sigma_f_num += pow((i - Mu_f), 2) * histogram[i]
        
        sigma_f = sigma_f_num / sum_h_k_f


        # Cálculo de sigma_w
        sigma_w = (Wb * sigma_b) + (Wf * sigma_f)

        if (sigma_w < x):
            x = sigma_w
            t2 = t1

    return t2


def binarize(image):
    h = histogram(image)
    threshold_value = otsu(h)

    binarized_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] < threshold_value:
                binarized_image[i,j] = 0
            else:
                binarized_image[i,j] = 255
    
    return binarized_image


def print_clusters(clf, img):
    images = {}
    for k, v in clf.items():
        base_img = np.zeros(img.shape, dtype=np.uint8)
        for coords in v:
            base_img[coords[0], coords[1]] = img[coords[0], coords[1]]        
        images[k] = base_img
    return images


def transform_contours(img, contours, selected):
    images = {}
    for i, con in enumerate(contours):
        coords = []
        for c in con:
            coords.append(tuple(c[0]))
        if i in selected:
            images[i] = coords
    return images


def middle_point(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def main():
    img_title = 'jitomate.png'
    image_rgb = Image.open(img_title)
    img_arr = np.array(image_rgb, dtype='int64')
    
    clf = clusterization(image_rgb)
    
    
    # # Guardamos los clusters en diferentes arreglos
    images = print_clusters(clf, img_arr)
    # for k, v in images.items():
    #     cv2.imshow(f"{k}", v)
    #     cv2.waitKey(0)

    cluster = images[3]
    # Guardamos en una imagen el cluster de nuestro interés (el que contiene los jotimates)
    im_cluster_0 = Image.fromarray(cluster).save("jitomates_clusterizados.png")
    cv2.imshow('Jitomates Clusterizados', cv2.cvtColor(cluster, cv2.COLOR_BGR2RGB))

    # la convertimos a escala de grises
    gray_image = np.array(Image.fromarray(cluster).convert('L'))
    cv2.imshow('Jitomates Clusterizados escala de grises', cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))

    # Guardamos la imagen en escala de grises
    Image.fromarray(gray_image).save("jitomates_clusterizados_gris.png")

    # Binarizamos la imagen
    binarized_image = binarize(gray_image)

    # Guardamos la imagen binarizada
    Image.fromarray(binarized_image).save("jitomates_binarizados.png")
    cv2.imshow('Jitomates Binarizados', cv2.cvtColor(binarized_image, cv2.COLOR_BGR2RGB))


    # Obtenemos los contornos con la imagen binarizada
    img_contours, hierarchy = cv2.findContours(binarized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(img_contours), hierarchy)

    result = cv2.imread('jitomate.png')

    # Obtenemos los contornos de las figuras de nuestro interés (los jitomates solicitados) en formato de opencv para imprimirlos
    selected_contours = [cont for i, cont in enumerate(img_contours) if i in [0, 4]]

    cv2.drawContours(image=result, contours=selected_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    only_contours = result.copy()

    cv2.drawContours(image=only_contours, contours=img_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('Solo contornos', only_contours)

    obj_int=[2, 4]
    for i, c in enumerate(selected_contours):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Obtenemos el rectángulo de area mínima
        rotrect = cv2.minAreaRect(c)
        # Obtenmos los puntos del dicho, para obtener los puntos iniciales y finales de las rectas a dibujar
        box_points = cv2.boxPoints(rotrect)
        # Conversion necesaria para dibujar las rectas
        box_points = np.int0(box_points)


        # Punto inicial (Punto con la y de valor mayor)
        p1 = box_points[0]
        # Calculamos y ordenamos las distancias euclidianas de menor a mayor
        dists_points = sorted([ (np.linalg.norm(p1 - box_points[j]), box_points[j]) for j in range(1, len(box_points)) ], key=lambda x: x[0])
        p2 = dists_points[0][1]
        p3 = dists_points[-2][1]
        p4 = dists_points[-1][1]
        print(f"Objeto {obj_int[i]} \n\tp1: {p1}, p2: {p2}, p3: {p3}, p4: {p4}")
        # Calculamos los puntos para trazar la línea
        start = middle_point(p1, p2)
        end = middle_point(p3, p4)
        print(f"\tstart: {start}, end: {end}")

        # Dibujamos el resultado final
        cv2.line(result, np.int0(start), np.int0(end), (255, 255, 255), 3)
        cv2.circle(result, (cX, cY), 7, (255, 255, 255), -1)
        cv2.drawContours(result,[box_points], 0, (0, 0, 255), 2)

    cv2.imshow("Resultado final", result)
    Image.fromarray( cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).save("Resultado final.png")
    cv2.waitKey(0)


if __name__ == "__main__":
    main()