import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


start_index = 4570
img_count = 1000
end_index = start_index + img_count
directory = (
    "/home/michael/Stuff/ffmpeg-tutorial/frames/lviv-2024-06-27/13:18/right_camera/"
)
prev_img = cv2.imread(f"{directory}/{start_index}.jpg")
# prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

total_time = 0
index = start_index

mov_count = 0
stand_count = 0

diff_values = [0]

while True:
    index += 1
    if index == end_index:
        break

    current_img = cv2.imread(f"{directory}/{index}.jpg")
    start_time = time.time()
    # current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    diff = np.mean(current_img - prev_img)

    end_time = time.time()
    if diff <= 10.0:
        diff_values.append(0)
        stand_count += 1
    else:
        diff_values.append(diff)
        mov_count += 1

    print(f"img: {index}, diff {diff}, time: {(end_time - start_time)*1000} ms")
    total_time += (end_time - start_time) * 1000
    prev_img = current_img
    cv2.imshow(f"img", current_img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
    elif key == 81:  # left arrow
        index -= 2
        prev_img = cv2.imread(f"{directory}/{index}.jpg")
        # prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    elif key == 83:  # right arrow
        prev_prev_img = prev_img
        prev_img = current_img


print(f"time: {total_time / (end_index - start_index)} ms")
print(
    f"mov: {mov_count}, stand: {stand_count}, ratio: {stand_count/(mov_count+stand_count)}"
)

indices = np.arange(img_count)
diff_values = np.array(diff_values)
print(indices.shape)
print(diff_values.shape)
# Plot a bar graph
plt.bar(indices, diff_values, color="blue", edgecolor="black")

# Add titles and labels
plt.title("Values Associated with Indices")
plt.xlabel("Index")
plt.xticks(np.arange(0, img_count, 200))  # Adjust the ticks to show every 200 units
plt.ylabel("Value")

# Show the plot
plt.show()

cv2.destroyAllWindows()
