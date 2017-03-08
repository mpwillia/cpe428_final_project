import numpy as np

class ConfusionMatrix():
   """
   Makes a confusion matrix array given a list of touples in the format 
      (expected, predicted)
   """
   def __init__(self, predictions, character_dict = None, num_classes = 63):
      self.character_dict = character_dict
      self.num_pred = len(predictions)
      self.confusion_array = np.zeros((num_classes, num_classes), dtype = np.int)

      for exp, pred in predictions:
         self.confusion_array[exp, pred] += 1;

   def get_array(self):
      return self.confusion_array

   def get_csv(self):
      translate_row = self.character_dict is not None
      csv = []
      header_row = [];
      header_row.append(' ')
      for row_num, row in enumerate(self.confusion_array):
         if translate_row:
            header_row.append(str(self.character_dict[row_num]))
         else:
            header_row.append(str(row_num))

      csv.append(', '.join(header_row))

      for row_num, row in enumerate(self.confusion_array):
         row_str = []
         if translate_row:
            row_str.append(str(self.character_dict[row_num]))
         else:
            row_str.append(str(row_num))
         
         for value in row:
            row_str.append(str(value))

         csv.append(', '.join(row_str))

      return '\n'.join(csv)

   def get_num_predictions(self):
      return self.num_pred
   
   def get_pos(self):
      total = 0
      for i in range(len(self.confusion_array)):
         total += self.confusion_array[i][i]

      return total

   def get_neg(self):
      return self.num_pred - self.get_pos() 

   def get_pos_percent(self):
      return self.get_pos() / self.num_pred

   def get_neg_percent(self):
      return self.get_neg() / self.num_pred

   def get_false_positives(self, pred_num):
      col = self.confusion_array[:, pred_num]
      num_false_pos = 0
      for idx, val in enumerate(col):
         if idx is not pred_num:
            num_false_pos += val

      return num_false_pos

   def get_false_negatives(self, exp_num):
      row = self.confusion_array[exp_num]
      num_false_neg = 0
      for idx, val in enumerate(row):
         if idx is not exp_num:
            num_false_neg += val

      return num_false_neg

if __name__ == '__main__':
   pred_array = ((0,0), (0,0), (0,0), (0,0), (0,0), (0,1), (1,0), (1,0), (1,1),
                  (1,1), (1,1), (1,2), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2))
   char_dict = {0:'a', 1:'b', 2:None}
   conf_mat = ConfusionMatrix(pred_array,character_dict = char_dict, num_classes = 3)
   print(conf_mat.get_false_negatives(0))
   print(conf_mat.get_false_negatives(1))
   print(conf_mat.get_false_negatives(2))
