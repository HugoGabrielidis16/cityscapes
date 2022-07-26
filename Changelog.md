# 25 juilllet 2022

1st commit :
Make visualisation function works. <br />
Training seems to works on CPU, have to test again on GPU.<br/>

2nd commit :
Tried to used tf_save to store my dataset but the final zip file is 513M while the dataset/data.zip is only 101M.
Since the preprocessing doesn't take that much times it is better to use data.zip-

# 26 juillet 20222

1st commit :
Adds of a pytorch version since sm.Unet doesn't work with GPU on tensorflow !
It seems to train on pytorch but the results are not so great -> looks into parameter optimization

2nd commit :
Correction of a minor bugs : didn't returned anythings on the test_loader function of the Module
