def calculate_top1_acc():
    pass

def calculate_top5_acc():
    pass


def main():
    import csv
    import xml.etree.cElementTree as ET
    
    with open('2024_2_16_16_33.csv') as csv_file:
        csv_reader = csv.reader(csv_file)

        
        number_of_images = 50000
        acc_5 = 0
        acc_1 = 0



        for row in csv_reader:
            if row[0] == "":
                continue

            is_highest_score = False
            highest_score = 0.0

            #parsing through the xml file
            image = row[1].split("/")[-1]
            image_name = image.split(".JPEG")[0]
            image_name = image_name + ".xml"
            tree = ET.parse(image_name)
            root = tree.getroot()
            for object in root.findall("object"):
                cls = object.find("name").text
                #print(cls)
                break

            with open('LOC_synset_mapping.txt') as f:
                lines = f.readlines()

            for line in lines:
                if cls in line:
                    name = line
                    break


            #print(image_name)
            
            var = 3

            while var+1 < len(row):
                #print([row[var], row[var+1], row[var+2]])
                #print(float(row[var+2]), highest_score)
                if float(row[var+2]) > highest_score:
                    highest_score = float(row[var+2])
                    highest_score_var = var+2
                    
                #print(row[var], line)
                if row[var] in name:
                    #print(row[var], row[var+1], row[var+2])
                    correct_label = row[var]
                    correct_label_var = var + 2

                    acc_5 = acc_5 + 1

                var = var +3
            

            print(correct_label_var, highest_score_var)
            if correct_label_var == highest_score_var:
                #print("hoy")
                #print(correct_label_var, highest_score_var)
                acc_1 = acc_1 + 1

                


                


                

    print(acc_5, number_of_images, acc_5/number_of_images)
    print(acc_1, number_of_images, acc_1/number_of_images)
            



if __name__ == "__main__":
    main()