from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
import numpy as np

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
detector = MTCNN()


def extrair_face(file_path, size=(160, 160)):
    img = Image.open(file_path)  # complete path
    img = img.convert('RGB')  # convert to RGB

    array = np.array(img)

    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']
    x2, y2, = x1 + width, y1 + height
    face = array[y1:y2, x1:x2]  # extrair sub matriz da image original

    image = Image.fromarray(face)
    image = image.resize(size)

    return image


def flip_image(image):
    img = image.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def load_fotos(directory_src, directory_target):
    for filename in listdir(directory_src):
        path_src = directory_src + filename
        path_target = directory_target + filename
        path_target_flip = directory_target + "flip-" + filename

        try:
            face = extrair_face(path_src)
            flip = flip_image(face)

            face.save(path_target, "JPEG", quality=100, optimize=True, progressive=True)
            flip.save(path_target_flip, "JPEG", quality=100, optimize=True, progressive=True)
        except:
            print("Erro na image {}".format(path_src))


def load_dir(directory_src, directory_target):
    for subir in listdir(directory_src):

        path = directory_src + subir + "\\"
        path_tg = directory_target + subir + "\\"

        if not isdir(path):
            continue

        load_fotos(path, path_tg)


if __name__ == '__main__':
    load_dir("C:\\Users\\rober\\Desktop\\python\\databases\\images\\",
             "C:\\Users\\rober\\Desktop\\python\\databases\\faces\\")
