# Aspectus
 ## Copyright (C) 2017  Aakash Sinha <aakash19developer@gmail.in>
 ## Credits : @warmspringwinds
 
#### We   will   be   performing   image   segmentaion   on   a   given   image.   The   machine 
#### learning   library   which   we   are   using   is   TensorFlow.   In   the   first   phase   we   used   Slim 
#### wrapper   and   VGG‐16   model   to   classify   an   image   in   over   1000   classes   and   give 
#### significant   probabilities   using   Softmax.   In   the   second   phase   we   have   trained 
#### FCN‐8s   net   on   VGG‐16   and   it   has   used   PASCAL   VOC   2012   model   (trained   on 
#### ImageNet   to   generate   21   classes)   to   generate   classes.   Then   we   have   used   CRF   as 
#### recursive   function   (in   RNN)   to   generate   a   heat   map   of   the   obtained   foreground 
#### which   have   been   classified.   Then   we   have   used   morphological   operations   on   the 
#### image   to   detect   contour   and   masking   to   retrieve   the   final   output   image.   Then   we 
#### have   extended   our   project   by   making   custom   stickers   for   Telegram   ChatBox.  
