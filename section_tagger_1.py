import os
from collections import defaultdict
section=['HISTORY OF PRESENT ILLNESS','Interval history','Past Medical History','Social History','Occupational History','Social History Main Topics','Family History','Current Outpatient Prescriptions',
         'REVIEW OF SYSTEMS','PHYSICAL EXAMINATION','NEUROLOGIC EXAMINATION','Review of imaging','ASSESSMENT AND PLAN','PLAN','Diagnoses and all orders for this visit']

def readfile(filename, d=defaultdict(list)):

    checkend=0
    checkIndex=0
    DicSections = d
    dicSentence = d
    with open(filename, 'r', ) as f:
        contents = f.read().replace('Â\xa0','').split("\n")
        print(contents)
        for v in contents:
            if v.lower().replace(':','')  == 'HISTORY OF PRESENT ILLNESS'.lower():
                index = contents.index(v)+1
                for vin in range(index,len(contents)-1):

                    if contents[vin].lower().replace(':','') not in map(lambda value: value.lower(), section) :       #map(lambda value: value.upper(), section )
                        if contents[vin] != '':
                            DicSections[section[0]].append(contents[vin])
                            checkIndex = contents.index(contents[vin])
                    else:
                        checkend=1
                        break

            elif v.lower().replace(':','') == 'Interval history'.lower() :
                index = contents.index(v)+1
                for vin2 in range (index, len(contents)-1):
                    if contents[vin2].lower().replace(':','') not in map(lambda value: value.lower(), section):
                        if contents[vin2] != '':
                            DicSections[section[1]].append(contents[vin2])
                            checkIndex = contents.index(contents[vin2])
                    else:
                        checkend=1
                        break

            elif v.lower().replace(':','') == 'Past Medical History'.lower():
                index =  contents.index(v)+1
                list=[]
                for vin3 in range(index, len(contents)-1):
                    if contents[vin3].lower().replace(':','') not in map(lambda value: value.lower(), section):
                        if contents[vin3] != '':
                            DicSections[section[2]].append(contents[vin3])
                            #list.append(contents[vin3])
                            checkIndex = contents.index(contents[vin3])
                            #checkend = 1

                    else:
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Social History'.lower():
                index =  contents.index(v)+1
                for vin4 in range(index, len(contents)-1):
                    if contents[vin4].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin4] != '':
                            DicSections[section[3]].append(contents[vin4])
                            checkIndex = contents.index(contents[vin4])
                            #checkend = 1

                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Occupational History'.lower():  #
                index =  contents.index(v)+1
                for vin5 in range(index, len(contents)-1):
                    if contents[vin5].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin5] != '':
                            DicSections[section[4]].append(contents[vin5])
                            checkIndex = contents.index(contents[vin5])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Social History Main Topics'.lower():
                index =  contents.index(v)+1
                for vin6 in range(index, len(contents)-1):
                    if contents[vin6].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin6] != '':
                            DicSections[section[5]].append(contents[vin6])
                            checkIndex = contents.index(contents[vin6])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Family History'.lower():
                index =  contents.index(v)+1
                for vin7 in range(index, len(contents)-1):
                    if contents[vin7].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin7] != '':
                            DicSections[section[6]].append(contents[vin7])
                            checkIndex = contents.index(contents[vin7])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Current Outpatient Prescriptions'.lower():
                index =  contents.index(v)+1
                for vin8 in range(index, len(contents)-1):
                    if contents[vin8].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin8] != '':
                            DicSections[section[7]].append(contents[vin8])
                            checkIndex = contents.index(contents[vin8])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'REVIEW OF SYSTEMS'.lower():
                index =  contents.index(v)+1
                for vin9 in range(index, len(contents)-1):
                    if contents[vin9].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin9] != '':
                            DicSections[section[8]].append(contents[vin9])
                            checkIndex = contents.index(contents[vin9])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'PHYSICAL EXAMINATION'.lower():
                index =  contents.index(v)+1
                for vin10 in range(index, len(contents)-1):
                    if contents[vin10].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin10] != '':
                            DicSections[section[9]].append(contents[vin10])
                            checkIndex = contents.index(contents[vin10])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'NEUROLOGIC EXAMINATION'.lower():
                index =  contents.index(v)+1
                for vin11 in range(index, len(contents)-1):
                    if contents[vin11].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin11] != '':
                            DicSections[section[10]].append(contents[vin11])
                            checkIndex = contents.index(contents[vin11])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Review of imaging'.lower():
                index =  contents.index(v)+1
                for vin12 in range(index, len(contents)-1):
                    if contents[vin12].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin12] != '':
                            DicSections[section[11]].append(contents[vin12])
                            checkIndex = contents.index(contents[vin12])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'ASSESSMENT AND PLAN'.lower():
                index =  contents.index(v)+1
                for vin13 in range(index, len(contents)-1):
                    if contents[vin13].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin13] != '':
                            DicSections[section[12]].append(contents[vin13])
                            checkIndex = contents.index(contents[vin13])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'PLAN'.lower():
                index =  contents.index(v)+1
                for vin14 in range(index, len(contents)-1):
                    if contents[vin14].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin14] != '':
                            DicSections[section[13]].append(contents[vin14])
                            checkIndex = contents.index(contents[vin14])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif v.lower().replace(':','') == 'Diagnoses and all orders for this visit'.lower():
                index =  contents.index(v)+1
                for vin15 in range(index, len(contents)-1):
                    if contents[vin15].lower().replace(':','') not in map(lambda value:value.lower(), section):
                        if contents[vin15] != '':
                            DicSections[section[14]].append(contents[vin15])
                            checkIndex = contents.index(contents[vin15])
                            #checkend = 1
                    else:
                        #continue
                        checkend = 1
                        break

            elif checkend == 0:
                #print("else index",index, "v is", v)
                for vin6 in range(checkIndex, len(contents)-1):
                    #print(contents[vin6])
                    if contents[vin6] != '':
                        dicSentence["sentences"].append(v)
                    else:
                        break
    if len(DicSections)>0:
        for k,v in DicSections.items():

            mlist = ' '.join(map(str, v))
            if k == 'HISTORY OF PRESENT ILLNESS' or k == 'Interval history' or k == 'Past Medical History' or k == 'Marital status' or k == 'Spouse name' or k == 'Number of children' or k == 'Years of education' or k == 'Smoking status' or k == 'Family History' or k == 'Current Outpatient Prescriptions' or k == 'REVIEW OF SYSTEMS':
                print(mlist, '\t', "Subjective" ,"\n")
            elif k == 'neurologic examination' or k == 'Review of imaging' or k == 'physical examination':
                print(mlist, '\t', "Objective" ,"\n" )

            elif k == 'assessment and plan':
                print(mlist, '\t', "Assessment" ,"\n")

            elif k == 'PLAN' or k == 'Diagnoses and all orders for this visit':
                print(mlist, '\t', "Plan" ,"\n")

if __name__ == '__main__':
    readfile("data/clinical_notes/note1.txt")








