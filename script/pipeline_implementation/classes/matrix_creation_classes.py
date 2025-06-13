import json
class matrix_creation():

    def __init__(self, chat, prompt):

        
        self.chat=chat
        
        self.prompt=prompt

        
    def calculate_matrix(self, triple):

        dict_result={}
        
        for i in triple.keys():
            
            for j in triple.keys():
                
                if not(i==j):
                    
                    '''print("################## PROMPT #####################")
                    #the first element of the dataset prompt
                    print(self.prompt.to_list()[0].format(triple[i], triple[j]))
                    print("################## RESPONSE #####################")
                    print("Pred")'''
                    
                    response = self.chat.send_prompt(self.prompt.to_list()[0].format(triple[i], triple[j]), prompt_uuid="1", use_history=False, stream=True)

                    '''print("\n")
                    print("True")
                    print("\n")'''
                    dict_result[(i,j)]=str(response.raw_text)

                else:
                    
                    dict_result[(i,j)]=str("X")
                    
        return dict_result



class triple_extraction():

    def __init__(self, file_path):

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.data=data

    def extract_list_tuple(self):

        grouped_by_id_prev = {}

        for i in self.data:

            for argument in i["arguments"]:

                grouped_by_id_prev.setdefault(argument["id_prev"], []).append(argument)
           
        return grouped_by_id_prev


