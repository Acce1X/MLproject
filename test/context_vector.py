#tfidf_file = open('./tfidf_file_with_default_idf.txt','r')
tfidf_file = open('./tfidf_file.txt','r')

tag_vector = []
line = tfidf_file.readline()
while line:
    words = line.split(' ')
    for word in words:
        if word not in tag_vector:
            tag_vector.append(word)
    line = tfidf_file.readline()
    

print(str(len(tag_vector))+'\n')
print(tag_vector)