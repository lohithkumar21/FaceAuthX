import os
directory = "C:/Users/lohit.DESKTOP-57KU0RV/OneDrive/Documents/vscode/Face_Anti_Spoofing/Dataset/DataCollect"

text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

#Handling missing coordinates in text file
for filename in text_files:
    text_file_path = os.path.join(directory, filename)
    image_file=filename.replace('.txt', '.jpg')
    image_file_path = os.path.join(directory,image_file)
    with open(text_file_path, 'r') as file:
        if file.read().strip() == "" and os.path.exists(image_file_path):  # Check if the text file is empty
            os.remove(image_file_path)  # Delete the corresponding image file
            print(f"Deleted: {image_file}")
                    
#Remove text files, if there is no associated image files
for text_file in text_files:
    image_file = text_file.replace('.txt', '.jpg')
    if not os.path.exists(os.path.join(directory, image_file)):
        os.remove(os.path.join(directory, text_file))
        print(f"Deleted {text_file}")
        
#Remove image files, if there is no associated text files       
image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
for image_file in image_files:
    text_file = image_file.replace('.jpg', '.txt')
    if not os.path.exists(os.path.join(directory, text_file)):
        os.remove(os.path.join(directory, image_file))
        print(f"Deleted {image_file}")
        
print("Files are removed successfully!")
