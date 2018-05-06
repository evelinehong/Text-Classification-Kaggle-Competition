from sklearn import preprocessing 
import csv
import json
import numpy as np
import pandas as pd

def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min);  
    return x;  
if __name__ == '__main__':
    val1=[]
    val2=[]
    val3=[]
    val4=[]
    val5=[]
    val6=[]
    val7=[]
    val8=[]
    val9=[]
    val10=[]
    val111=[]
    val12=[]
    val13=[]
    val14=[]
    val15=[]
    val16=[]
    val17=[]
    val18=[]
    val19=[]
    val20=[]
    val21=[]
    val222=[]
    val23=[]
    val24=[]
    val25=[]
    val26=[]
    val27=[]
    val28=[]
    val29=[]
    val30=[]
    val31=[]
    val32=[]
    val333=[]
    val34=[]
    val35=[]
    val36=[]
    val37=[]
    val38=[]
    val39=[]
    val40=[]
    val41=[]
    val42=[]
    val43=[]
    val444=[]
    val45=[]
    val46=[]
    val47=[]
    val48=[]
    val49=[]
    val50=[]
    val51=[]
    val52=[]
    val53=[]
    val54=[]
    val555=[]
    val56=[]
    val57=[]
    val58=[]
    val59=[]
    val60=[]
    val61=[]


    with open('chinese/predict_LR5.csv', 'r') as f3:
        reader1 = csv.reader(f3)
        for row in reader1:
            if row[1] != 'pred':
               val1.append(float(row[1]))
    val11=pd.Series(val1)

  
    filename = 'prob-ck2000-lr0001.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] != 'pred':
                val2.append(float(row[1]))
    val22=pd.Series(val2)

    filename2 = 'prob-ck13000-lr001.csv'
    with open(filename2) as f5:
        reader2 = csv.reader(f5)
        for row in reader2:
            if row[1] != 'pred':
                val3.append(float(row[1]))
    val33=pd.Series(val3)

    filename3 = 'prob-ck12500-lr001.csv'
    with open(filename3) as f6:
        reader3 = csv.reader(f6)
        for row in reader3:
            if row[1] != 'pred':
                val4.append(float(row[1]))
    val44=pd.Series(val4)

    # filename4 = 'prob-ck10500-lr001.csv'
    # with open(filename4) as f7:
    #     reader4 = csv.reader(f7)
    #     for row in reader4:
    #         if row[1] != 'pred':
    #             pred7.append(row[1])
    # pred7_scale = preprocessing.scale(pred7)

    filename5 = 'prob-ck18500-lr0001-L20.4.csv'
    with open(filename5) as f8:
        reader5 = csv.reader(f8)
        for row in reader5:
            if row[1] != 'pred':
                val5.append(float(row[1]))
    val55=pd.Series(val5)

    filename6 = 'prob-ck11500-lr0001-L20.4.csv'
    with open(filename6) as f9:
        reader6 = csv.reader(f9)
        for row in reader6:
            if row[1] != 'pred':
                val6.append(float(row[1]))
    val66=pd.Series(val6)

    filename7 = 'prob-ck10500-lr0001-L20.4.csv'
    with open(filename7) as f10:
        reader7 = csv.reader(f10)
        for row in reader7:
            if row[1] != 'pred':
                val7.append(float(row[1]))
    val77=pd.Series(val7)

    filename8 = 'prob-ck11500-lr00001-1524290099.csv'
    with open(filename8) as f11:
        reader8 = csv.reader(f11)
        for row in reader8:
            if row[1] != 'pred':
                val8.append(float(row[1]))
    val88=pd.Series(val8)

    filename9 = 'prob-ck11000-lr0001-1524290099.csv'
    with open(filename9) as f12:
        reader9 = csv.reader(f12)
        for row in reader9:
            if row[1] != 'pred':
                val9.append(float(row[1]))
    val99=pd.Series(val9)

    filename10 = 'prob-ck7000-lr0001-1524290099.csv'
    with open(filename10) as f13:
        reader10 = csv.reader(f13)
        for row in reader10:
            if row[1] != 'pred':
                val10.append(float(row[1]))
    val1010=pd.Series(val10)

    filename11 = 'prob-ck17500-lr00001-1524368649.csv'
    with open(filename11) as f14:
        reader11 = csv.reader(f14)
        for row in reader11:
            if row[1] != 'pred':
                val111.append(float(row[1]))
    val1111=pd.Series(val111)

    filename12 = 'prob-ck11500-lr00001-1524368649.csv'
    with open(filename12) as f15:
        reader12 = csv.reader(f15)
        for row in reader12:
            if row[1] != 'pred':
                val12.append(float(row[1]))
    val1212=pd.Series(val12)
    val1111=pd.Series(val12)

    filename13 = 'prob-ck8500-lr00001-1524368649.csv'
    with open(filename13) as f116:
        reader13 = csv.reader(f116)
        for row in reader13:
            if row[1] != 'pred':
                val13.append(float(row[1]))
    val1313=pd.Series(val13)

    filename14 = 'prob-ck6500-lr00001-1524368649.csv'
    with open(filename14) as f117:
        reader14 = csv.reader(f117)
        for row in reader14:
            if row[1] != 'pred':
                val14.append(float(row[1]))
    val1414=pd.Series(val14)

    filename15 = 'prob-ck7000-lr00001-1524450365.csv'
    with open(filename15) as f118:
        reader15 = csv.reader(f118)
        for row in reader15:
            if row[1] != 'pred':
                val15.append(float(row[1]))
    val1515=pd.Series(val15)

    filename16 = 'prob-ck3500-lr00001-1524450365.csv'
    with open(filename16) as f119:
        reader16 = csv.reader(f119)
        for row in reader16:
            if row[1] != 'pred':
                val16.append(float(row[1]))
    val1616=pd.Series(val16)

    filename17 = 'prob-ck5500-lr00001-1524450365.csv'
    with open(filename17) as f120:
        reader17 = csv.reader(f120)
        for row in reader17:
            if row[1] != 'pred':
                val17.append(float(row[1]))
    val1717=pd.Series(val17)

    filename18 = 'prob-ck5000-lr00001-1524450365.csv'
    with open(filename18) as f121:
        reader18 = csv.reader(f121)
        for row in reader18:
            if row[1] != 'pred':
                val18.append(float(row[1]))
    val1818=pd.Series(val18)

    filename19 = 'prob-ck7000-lr00001-1524497893.csv'
    with open(filename19) as f121:
        reader19 = csv.reader(f121)
        for row in reader19:
            if row[1] != 'pred':
                val19.append(float(row[1]))
    val1919=pd.Series(val19)

    filename20 = 'prob-ck9000-lr00001-1524497893.csv'
    with open(filename20) as f122:
        reader20 = csv.reader(f122)
        for row in reader20:
            if row[1] != 'pred':
                val20.append(float(row[1]))
    val2020=pd.Series(val20)

    filename21 = 'prob-ck11000-lr00001-1524497893.csv'
    with open(filename18) as f123:
        reader21 = csv.reader(f123)
        for row in reader21:
            if row[1] != 'pred':
                val21.append(float(row[1]))
    val2121=pd.Series(val21)

    filename22 = 'prob-ck5500-lr00001-1524497893.csv'
    with open(filename18) as f124:
        reader22 = csv.reader(f124)
        for row in reader22:
            if row[1] != 'pred':
                val222.append(float(row[1]))
    val2222=pd.Series(val222)

    with open('chinese/predict_LR.csv', 'r') as f25:
        reader23 = csv.reader(f25)
        for row in reader23:
            if row[1] != 'pred':
               val23.append(float(row[1]))
    val2323=pd.Series(val23)

    with open('chinese/predict_LR2.csv', 'r') as f26:
        reader24 = csv.reader(f26)
        for row in reader24:
            if row[1] != 'pred':
               val24.append(float(row[1]))
    val2424=pd.Series(val24)

    with open('chinese/predict_LR3.csv', 'r') as f27:
        reader25 = csv.reader(f27)
        for row in reader25:
            if row[1] != 'pred':
               val25.append(float(row[1]))
    val2525=pd.Series(val25)

    with open('chinese/predict_LR4.csv', 'r') as f28:
        reader26 = csv.reader(f28)
        for row in reader26:
            if row[1] != 'pred':
               val26.append(float(row[1]))
    val2626=pd.Series(val26)

    with open('prob-ck3500-lr00001-1524653628.csv', 'r') as f29:
        reader27 = csv.reader(f29)
        for row in reader27:
            if row[1] != 'pred':
               val27.append(float(row[1]))
    val2727=pd.Series(val27)

    with open('prob-ck5000-lr00001-1524653628.csv', 'r') as f30:
        reader28 = csv.reader(f30)
        for row in reader28:
            if row[1] != 'pred':
               val28.append(float(row[1]))
    val2828=pd.Series(val28)

    with open('prob-ck6000-lr00001-1524653628.csv', 'r') as f31:
        reader29 = csv.reader(f31)
        for row in reader29:
            if row[1] != 'pred':
               val29.append(float(row[1]))
    val2929=pd.Series(val29)

    with open('mxgb-rank.csv', 'r') as f32:
        reader30 = csv.reader(f32)
        for row in reader30:
            if row[1] != 'pred':
               val30.append(float(row[1]))
    val3030=pd.Series(val30)

    with open('prob-ck8000-lr00001-1524653628.csv', 'r') as f33:
        reader31 = csv.reader(f33)
        for row in reader31:
            if row[1] != 'pred':
               val31.append(float(row[1]))
    val3131=pd.Series(val31)

    with open('prob-ck3500-1524672748.csv', 'r') as f34:
        reader32 = csv.reader(f34)
        for row in reader32:
            if row[1] != 'pred':
               val32.append(float(row[1]))
    val3232=pd.Series(val32)

    with open('prob-ck11000-1524672748.csv', 'r') as f35:
        reader33 = csv.reader(f35)
        for row in reader33:
            if row[1] != 'pred':
               val333.append(float(row[1]))
    val3333=pd.Series(val333)

    with open('prob-ck5000-1524672748.csv', 'r') as f36:
        reader34 = csv.reader(f36)
        for row in reader34:
            if row[1] != 'pred':
               val34.append(float(row[1]))
    val3434=pd.Series(val34)

    with open('chinese/predict_LR.csv', 'r') as f37:
        reader35 = csv.reader(f37)
        for row in reader35:
            if row[1] != 'pred':
               val35.append(float(row[1]))
    val3535=pd.Series(val35)

    with open('submission.csv', 'r') as f38:
        reader36 = csv.reader(f38)
        for row in reader36:
            if row[1] != 'pred':
               val36.append(float(row[1]))
    val3636=pd.Series(val36)

    with open('prob-ck3500-1524973805.csv', 'r') as f39:
        reader37 = csv.reader(f39)
        for row in reader37:
            if row[1] != 'pred':
               val37.append(float(row[1]))
    val3737=pd.Series(val37)

    with open('prob-ck6000-1524973805.csv', 'r') as f40:
        reader38 = csv.reader(f40)
        for row in reader38:
            if row[1] != 'pred':
               val38.append(float(row[1]))
    val3838=pd.Series(val38)

    with open('prob-ck7000-1524973805.csv', 'r') as f41:
        reader39 = csv.reader(f41)
        for row in reader39:
            if row[1] != 'pred':
               val39.append(float(row[1]))
    val3939=pd.Series(val39)

    with open('predict_LR_0.85C.csv', 'r') as f42:
        reader40 = csv.reader(f42)
        for row in reader40:
            if row[1] != 'pred':
               val40.append(float(row[1]))
    val4040=pd.Series(val40)

    with open('predict_LR_0.95C.csv', 'r') as f43:
        reader41 = csv.reader(f43)
        for row in reader41:
            if row[1] != 'pred':
               val41.append(float(row[1]))
    val4141=pd.Series(val41)

    with open('prob-ck11500-1524973805.csv', 'r') as f44:
        reader42 = csv.reader(f44)
        for row in reader42:
            if row[1] != 'pred':
               val42.append(float(row[1]))
    val4242=pd.Series(val42)

    with open('prob-ck7500-1525056279.csv', 'r') as f45:
        reader43 = csv.reader(f45)
        for row in reader43:
            if row[1] != 'pred':
               val43.append(float(row[1]))
    val4343=pd.Series(val43)

    with open('prob-ck3500-1525056279.csv', 'r') as f46:
        reader44 = csv.reader(f46)
        for row in reader44:
            if row[1] != 'pred':
               val444.append(float(row[1]))
    val4444=pd.Series(val444)

    with open('prob-ck2000-1525056279.csv', 'r') as f47:
        reader45 = csv.reader(f47)
        for row in reader45:
            if row[1] != 'pred':
               val45.append(float(row[1]))
    val4545=pd.Series(val45)

    with open('prob-ck5500-1525142407.csv', 'r') as f48:
        reader46 = csv.reader(f48)
        for row in reader46:
            if row[1] != 'pred':
               val46.append(float(row[1]))
    val4646=pd.Series(val46)

    with open('prob-ck3500-1525142407.csv', 'r') as f49:
        reader47 = csv.reader(f49)
        for row in reader47:
            if row[1] != 'pred':
               val47.append(float(row[1]))
    val4747=pd.Series(val47)

    with open('prob-ck7000-1525142407.csv', 'r') as f50:
        reader48 = csv.reader(f50)
        for row in reader48:
            if row[1] != 'pred':
               val48.append(float(row[1]))
    val4848=pd.Series(val48)

    with open('prob-ck8000-1525142407.csv', 'r') as f51:
        reader49 = csv.reader(f51)
        for row in reader49:
            if row[1] != 'pred':
               val49.append(float(row[1]))
    val4949=pd.Series(val49)

    with open('prob-ck6000-1525254905.csv', 'r') as f52:
        reader51 = csv.reader(f52)
        for row in reader51:
            if row[1] != 'pred':
               val51.append(float(row[1]))
    val5151=pd.Series(val51)

    with open('prob-ck3500-1525254905.csv', 'r') as f53:
        reader52 = csv.reader(f53)
        for row in reader52:
            if row[1] != 'pred':
               val52.append(float(row[1]))
    val5252=pd.Series(val52)

    with open('prob-ck4000-1525254905.csv', 'r') as f54:
        reader53 = csv.reader(f54)
        for row in reader53:
            if row[1] != 'pred':
               val53.append(float(row[1]))
    val5353=pd.Series(val53)

    with open('final.csv', 'r') as f55:
        reader54 = csv.reader(f55)
        for row in reader54:
            if row[1] != 'pred':
               val54.append(float(row[1]))
    val5454=pd.Series(val54)

    with open('prob-ck2000-1525310627.csv', 'r') as f56:
        reader55 = csv.reader(f56)
        for row in reader55:
            if row[1] != 'pred':
               val555.append(float(row[1]))
    val5555=pd.Series(val555)

    with open('prob-ck3500-1525310627.csv', 'r') as f57:
        reader56 = csv.reader(f57)
        for row in reader56:
            if row[1] != 'pred':
               val56.append(float(row[1]))
    val5656=pd.Series(val56)

    with open('prob-ck5500-1525310627.csv', 'r') as f58:
        reader57 = csv.reader(f58)
        for row in reader57:
            if row[1] != 'pred':
               val57.append(float(row[1]))
    val5757=pd.Series(val57)

    with open('prob-ck7000-1525310627.csv', 'r') as f59:
        reader58 = csv.reader(f59)
        for row in reader58:
            if row[1] != 'pred':
               val58.append(float(row[1]))
    val5858=pd.Series(val58)

    with open('prob-ck8000-1525310627.csv', 'r') as f60:
        reader59 = csv.reader(f60)
        for row in reader59:
            if row[1] != 'pred':
               val52.append(float(row[1]))
    val5959=pd.Series(val59)

    with open('prob-ck3500-1525338517.csv', 'r') as f61:
        reader60 = csv.reader(f61)
        for row in reader60:
            if row[1] != 'pred':
               val60.append(float(row[1]))
    val6060=pd.Series(val60)

    with open('prob-ck6000-1525338517.csv', 'r') as f62:
        reader61 = csv.reader(f62)
        for row in reader61:
            if row[1] != 'pred':
               val61.append(float(row[1]))
    val6161=pd.Series(val61)

    pred1 = 1.0/(val11.rank())
    pred2 = 1.0/(val22.rank())
    pred3 = 1.0/(val33.rank())
    pred4 = 1.0/(val44.rank())
    pred5 = 1.0/(val55.rank())
    pred6 = 1.0/(val66.rank())
    pred7 = 1.0/(val77.rank())
    pred8 = 1.0/(val88.rank())
    pred9 = 1.0/(val99.rank())
    pred10 = 1.0/(val1010.rank())
    pred11 = 1.0/(val1111.rank())
    pred12 = 1.0/(val1212.rank())
    pred13 = 1.0/(val1313.rank())
    pred14 = 1.0/(val1414.rank())
    pred15 = 1.0/(val1515.rank())
    pred16 = 1.0/(val1616.rank())
    pred17 =  1.0/(val1717.rank())
    pred18 = 1.0/ (val1818.rank())
    pred19 = 1.0/ (val1919.rank())
    pred20 = 1.0/ (val2020.rank())
    pred21 = 1.0/ (val2121.rank())
    pred22 = 1.0/ (val2222.rank())
    pred23 = 1.0/ (val2323.rank())
    pred24 = 1.0/ (val2424.rank())
    pred25 = 1.0/ (val2525.rank())
    pred26 = 1.0/ (val2626.rank())
    pred27 = 1.0/ (val2727.rank())
    pred28 = 1.0/ (val2828.rank())
    pred29 = 1.0/ (val2929.rank())
    pred30 = 1.0/ (val3030.rank())
    pred31 = 1.0/ (val3131.rank())
    pred32 = 1.0/ (val3232.rank())
    pred33 = 1.0/ (val3333.rank())
    pred34 = 1.0/ (val3434.rank())
    pred36 = 1.0/ (val3636.rank())
    pred37 = 1.0/ (val3737.rank())
    pred38 = 1.0/ (val3838.rank())
    pred39 = 1.0/ (val3939.rank())
    pred40 = 1.0/ (val4040.rank())
    pred41 = 1.0/ (val4141.rank())
    pred42 = 1.0/ (val4242.rank())
    pred43 = 1.0/ (val4343.rank())
    pred44 = 1.0/ (val4444.rank())
    pred45 = 1.0/ (val4545.rank())
    pred46 = 1.0/ (val4646.rank())
    pred47 = 1.0/ (val4747.rank())
    pred48 = 1.0/ (val4848.rank())
    pred49 = 1.0/ (val4949.rank())

    pred51 =1.0/(val5151.rank())
    pred52 = 1.0/ (val5252.rank())
    pred53 = 1.0/ (val5353.rank())
    pred54 = 1.0/ ( val5454.rank())

    pred55 =1.0/(val5555.rank())
    pred56 = 1.0/ (val5656.rank())
    pred57 = 1.0/ ( val5757.rank())

    pred58 =1.0/(val5858.rank())
    pred59 = 1.0/ (val5959.rank())
    pred60 = 1.0/ (val6060.rank())
    pred61 = 1.0/ (val6161.rank())


    Id = []
    with open('chinese/predict_LR5.csv', 'r') as f3:
        reader1 = csv.reader(f3)
        for row in reader1:
            if row[1] != 'pred':
                Id.append(row[0])
    predn = (pred55 + pred56 + pred57 + pred58 + pred59)/5
    predo = (pred60 + pred61)/2
    preda = (pred4 + pred2 + pred3)/3
    predb = (pred6 + pred7 + pred5)/3
    predc = (pred8 + pred9 + pred10)/3
    predd = (pred11 + pred12 + pred13 + pred14)/4
    prede = (pred15 + pred16 + pred17 +pred18)/4
    predf = (pred19 + pred20 + pred21 +pred22)/4
    predg = (pred23 + pred24 + pred25 +pred26)/4 * 0.2 + 0.2 * pred1 + 0.3*pred40 + 0.3*pred41
    predh = (pred27 + pred28 + pred29 + pred31)/4
    predi = (pred32 + pred33 + pred34)/3
    predj = (pred37 + pred38 + pred39 + pred42)/4
    predk = (pred43 + pred44 + pred45)/3
    predl = (pred47 + pred48 + pred49 + pred46)/4
    predm = (pred51 + pred52 + pred53)/3

    pred = 0.023* predd + 0.004 * predm + 0.006 * preda + predb * 0.009+  0.009 * predc + 0.009 * predi + 0.50 * pred30 + 0.04 * predg + 0.055* prede + 0.055*predf + 0.055* predh + 0.055 * predj + 0.055 * predk + 0.055*predl + predo * 0.03 + 0.04 * predn
    
    pred = pred * 0.93 + 0.07 * pred36
    # predall = pred36
    # predxgb = predall - (0.04* predd + 0.010 * preda + 0.015 * predb + 0.015 * predc + 0.01 * predi + 0.20 * predg + 0.09* prede + 0.095*predf + 0.095* predh )
    # predxgb = predxgb/0.43
    # pred = pred36 * 0.9 + pred30 * 0.1
    # pred = (40239-pred)/40239


    filename2 = 'mean-rank53-10.csv'
    with open (filename2,'w') as f2:
        writer = csv.writer(f2,lineterminator='\n')
        prob = np.column_stack((Id, pred))
        writer.writerows([['id', 'pred']])
        writer.writerows(prob)