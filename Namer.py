import os

q = 0


for root, dirs, files in os.walk("C:\\Users\\Zippy\\OneDrive\\Neural Networks\\HandDigets\\DigetsByMe"):
        for file in files:
            if file.endswith('.png'):
                q = q+1
                print(os.path.join(root, file))
                old_path = os.path.join(root, file)
                new_path = f"{root}\diget{q}.png"
                print(new_path)
                os.rename(old_path, new_path)
