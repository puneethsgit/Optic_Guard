from tkinter import *
from tkinter import ttk, filedialog
from tkinter.ttk import *
from PIL import Image, ImageEnhance, ImageTk
import cv2
import numpy as np
from keras.models import load_model # type: ignore
from twilio.rest import Client
from playsound import playsound  # Import playsound library

class GlaucomaDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Optic Guard Glaucoma Detection")
        master.geometry("1200x500")

        # Schedule the playback of the sound clip after 3 seconds
        master.after(5000, self.play_sound)

        self.background_images = [
            'D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\gui.jpeg',
            'D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\gui2.jpeg'
        ]
        self.current_image_index = 0
        self.background_label = Label(master)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.update_background_image()

        self.heading_label = ttk.Label(master, text="Optic Guard Glaucoma Detection", style='Heading.TLabel')
        self.heading_label.place(relx=0.5, rely=0, anchor='n')

        style = ttk.Style()
        style.configure('TButton', font=('calibri', 15), foreground='black', background='white')  

        self.btn = Button(master, text="Select an image", command=self.select_image, style='TButton')
        self.btn.pack(side="bottom", expand="no", padx="10", pady="10")

        self.btn_gray = Button(master, text="Gray", command=self.gray, style='TButton')
        self.btn_gray.pack(side="bottom", expand="no", padx="10", pady="10")

        self.btn_ip = Button(master, text="Image Processing", command=self.ip, style='TButton')
        self.btn_ip.pack(side="bottom", expand="no", padx="10", pady="10")

        self.btn_ml = Button(master, text="Machine Learning + IP", command=self.ml, style='TButton')
        self.btn_ml.pack(side="bottom", expand="no", padx="10", pady="10")

        self.update_background_image_loop()

        self.panelA = None
        self.panelB = None
        self.panelC = None
        self.panelD = None
        self.prashu = None
        self.Ip = 0  

    def update_background_image(self):
        image_path = self.background_images[self.current_image_index]
        image = Image.open(image_path)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.6)
        image = image.resize((self.master.winfo_width(), self.master.winfo_height()), Image.LANCZOS)  
        photo = ImageTk.PhotoImage(image)
        self.background_label.configure(image=photo)
        self.background_label.image = photo  

    def update_background_image_loop(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.background_images)
        self.update_background_image()
        self.master.after(3000, self.update_background_image_loop)

    def play_sound(self):
        # Play the sound clip
        playsound('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\sound.mp3')

    def gray(self):
        if self.prashu is not None:
            gray = cv2.cvtColor(self.prashu, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(gray)
            image = ImageTk.PhotoImage(image)
            if self.panelD is None:
                self.panelD = Label(image=image)
                self.panelD.image = image
                self.panelD.pack(side="left", padx=10, pady=10)
            else:
                self.panelD.configure(image=image)
                self.panelD.image = image

    def select_image(self):
        path = filedialog.askopenfilename()
        if len(path) > 0:
            image = cv2.imread(path)
            self.prashu = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            if self.panelA is None:
                self.panelA = Label(image=image)
                self.panelA.image = image
                self.panelA.pack(side="left", padx=10, pady=10)
            else:
                self.panelA.configure(image=image)
                self.panelA.image = image

    def ip(self):
        if self.prashu is not None:
            image = self.prashu.copy()
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            orig = image.copy()  
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

            cv2.circle(image, maxLoc, 80, (0, 0, 0), 2)
            disc = 3.14 * 80 * 80

            r, g, b = cv2.split(orig)

            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(g, kernel, iterations=1)

            minmax_img = np.zeros((img_dilation.shape[0], img_dilation.shape[1]), dtype='uint8')

            for i in range(img_dilation.shape[0]):
                for j in range(img_dilation.shape[1]):
                    minmax_img[i, j] = 255 * (img_dilation[i, j] - np.min(img_dilation)) / (
                                np.max(img_dilation) - np.min(img_dilation))

            merge = cv2.merge((r, minmax_img, b))

            HSV_img = cv2.cvtColor(merge, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(HSV_img)

            median = cv2.medianBlur(s, 5)
            merge1 = cv2.merge((h, s, median))

            cv2.imwrite('merge_oc.jpg', merge1)
            image_merge = Image.open('merge_oc.jpg')

            enh_col = ImageEnhance.Color(image_merge)
            image_colored_oc = enh_col.enhance(7)

            cv2.imwrite('image_colored_oc.jpg', np.float32(image_colored_oc))
            image_c_oc = cv2.imread('image_colored_oc.jpg')

            lab = cv2.cvtColor(image_c_oc, cv2.COLOR_BGR2LAB)

            Z = lab.reshape((-1, 3))
            Z = np.float32(Z)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            K = 2
            ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center1 = np.uint8(center1)
            res1 = center1[label1.flatten()]
            output1 = res1.reshape((lab.shape))

            bilateral_filtered_image = cv2.bilateralFilter(output1, 5, 175, 175)
            edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

            contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_list = []
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                area = cv2.contourArea(contour)
                if ((len(approx) > 8) & (area > 30)):
                    contour_list.append(contour)
            cv2.drawContours(image, contour_list, -1, (255, 0, 0), 1)

            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(image, ellipse, (0, 0, 0), 1, cv2.LINE_AA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            if self.panelC is None:
                self.panelC = Label(image=image)
                self.panelC.image = image
                self.panelC.pack(side="left", padx=10, pady=10)
            else:
                self.panelC.configure(image=image)
                self.panelC.image = image

            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            cuparea = (3.14 / 3) * MA * ma

            cdr = cuparea / disc

            if cdr > 0.5:
                self.Ip = 1
                text = Text(self.master, height=6, font=('calibri', 13), foreground='black')
                text.insert(INSERT, "IMAGE PROCESSING RESULTS\n")
                text.insert(INSERT, "Area of Disc : " + str(disc) + "\n")
                text.insert(INSERT, "Area of Cup : " + str(cuparea) + "\n")
                text.insert(INSERT, "Cup to Disc Ratio : " + str(cdr) + "\n")
                text.insert(INSERT, "GLAUCOMA")
                text.place(relx=0.5, rely=0.05, anchor='n')
            else:
                self.Ip = 0
                text = Text(self.master, height=6, font=('calibri', 13), foreground='black')
                text.insert(INSERT, "IMAGE PROCESSING RESULTS\n")
                text.insert(INSERT, "Area of Disc : " + str(disc) + "\n")
                text.insert(INSERT, "Area of Cup : " + str(cuparea) + "\n")
                text.insert(INSERT, "Cup to Disc Ratio : " + str(cdr) + "\n")
                text.insert(INSERT, "NORMAL")
                text.place(relx=0.5, rely=0.05, anchor='n')

    def ml(self):
        print(self.Ip)
        model = load_model('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\glaucoma.h5')
        print("model loaded")
        cv2.imwrite('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\working1\\prashu.jpg', self.prashu)
        test_image = Image.open('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\working1\\prashu.jpg').resize((240, 240))
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        if result[0][0] != 1:
            DL = 1  
        else:
            DL = 0  

        if (DL and self.Ip) == 1:
            print("{} and {}".format(str(DL), str(self.Ip)))
            print("glaucoma")
            text = Text(self.master, height=9, font=('calibri', 13), foreground='black')
            text.insert(INSERT, "MACHINE LEARNING AND IMAGE PROCESSING\n")
            text.insert(INSERT, "Model Loaded\n")
            text.insert(INSERT, "GLAUCOMA")
            text.place(relx=0.5, rely=0.5, anchor='n')
            account_sid = ''
            auth_token = ''
            client = Client(account_sid, auth_token)
            message = client.messages \
                .create(
                    body="You are Affected by Glaucoma",
                    from_='+12513091659',
                    to='+918296986769'
                )
            print(message.sid)
        elif (DL and self.Ip) == 0:
            print("{} and {}".format(str(DL), str(self.Ip)))
            print("normal")
            text = Text(self.master, height=9, font=('calibri', 13), foreground='black')
            text.insert(INSERT, "MACHINE LEARNING AND IMAGE PROCESSING\n")
            text.insert(INSERT, "Model Loaded\n")
            text.insert(INSERT, "NORMAL")
            text.place(relx=0.5, rely=0.5, anchor='n')
            account_sid = ''
            auth_token = ''
            client = Client(account_sid, auth_token)
            message = client.messages \
                .create(
                body="You are Normal",
                from_='+12513091659',#+12513091659
                to='+918296986769'#+918296986769
            )
            print(message.sid)
        else:
            print("{} and {}".format(str(DL), str(self.Ip)))
            print("suspect")
            text = Text(self.master, height=9, font=('calibri', 13), foreground='black')
            text.insert(INSERT, "MACHINE LEARNING AND IMAGE PROCESSING\n")
            text.insert(INSERT, "Model Loaded\n")
            text.insert(INSERT, "SUSPECT")
            text.place(relx=0.5, rely=0.5, anchor='n')
            account_sid = ''
            auth_token = ''
            client = Client(account_sid, auth_token)
            message = client.messages \
                .create(
                body="You are a Suspect",
                from_='+12513091659',#+12513091659
                to='+918296986769'#+918296986769
            )
            print(message.sid)
        k = cv2.imread('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\working1\\Accuracy.jpeg')
        cv2.imshow('Model_Accuracy', k)
        j = cv2.imread('D:\\Edu\\projects\\Glucoma Detetection\\2024\\Final project\\working1\\Loss.jpeg')
        cv2.imshow('Loss', j)

root = Tk()
app = GlaucomaDetectionApp(root)
root.mainloop()
