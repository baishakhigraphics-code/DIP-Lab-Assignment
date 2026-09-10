import cv2
import matplotlib.pyplot as plt

# 1. Read the image
image1 = cv2.imread('images.jpg')

# Check if image is loaded successfully
if image1 is None:
    print("Error: images.jpg not found!")
    print("Make sure images.jpg and dip.py are in the same folder.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


# 2. Display the original image
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Original Image')
plt.show()


# 3. Print image dimensions
print('Image shape:', image_rgb.shape)


# 4. Print pixel value at (100,100)
pixel_value = image1[100, 100]
print('Pixel value at (100,100):', pixel_value)


# 5. Convert image to grayscale
gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


# 6. Print grayscale image dimensions
print('Grayscale Image shape:', gray_image.shape)


# 7. Display grayscale image
plt.imshow(gray_image, cmap='gray')
plt.axis('off')
plt.title('Gray Scale Image')
plt.show()


# 8. Resize the image to 300 x 300
resize = cv2.resize(image_rgb, (300, 300))

plt.imshow(resize)
plt.axis('off')
plt.title('Resized Image (300 x 300)')
plt.show()


# 9. Crop a portion of the image
crop = image_rgb[100:400, 100:400]

plt.imshow(crop)
plt.axis('off')
plt.title('Cropped Image')
plt.show()


# 10. Save the grayscale image
cv2.imwrite('gray_output.jpg', gray_image)

print('Grayscale image saved as gray_output.jpg')