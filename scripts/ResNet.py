from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense

def basic_block(x, filters, stride=1):
    res = Conv1D(filters, kernel_size=16, strides=stride, padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('relu')(res)
    res = Conv1D(filters, kernel_size=8, strides=1, padding='same')(res)
    res = BatchNormalization()(res)
    if stride != 1 or x.shape[-1] != filters:
        x = Conv1D(filters, kernel_size=8, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
    out = Add()([res, x])
    out = Activation('sigmoid')(out)
    return out

def build_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=16, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = basic_block(x, filters=32)
    x = basic_block(x, filters=32)
    x = basic_block(x, filters=64, stride=4)
    x = basic_block(x, filters=64)
    x = basic_block(x, filters=64, stride=4)
    x = basic_block(x, filters=64)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def Prepare_Data(FN1,FN2,INX):
    PG=open(FN1,'r')
    NG=open(FN2,'r')
    DT1=PG.readlines()
    DT2=NG.readlines()
    PG.close()
    NG.close()
    TRDT=[]
    TEDT=[]
    TRDL=[]
    TEDL=[]
    SL=list(range(0,len(DT1)))
    for i in range(0,round(len(DT1)*0.8)):
        r=np.random.randint(len(SL))
        del SL[r]
        k=DT1[r].split(',')
        ls=[]
        for j in range(0,len(k)):
            ls.append(float(k[j]))
        if len(ls)==((148+INX*2)*4):
            TRDT.append(ls)
            TRDL.append(1)
    for i in range(0,len(SL)):
        k=DT1[SL[i]].split(',')
        ls=[]
        for j in range(0,len(k)):
            ls.append(float(k[j]))
        if len(ls)==((148+INX*2)*4):
            TEDT.append(ls)
            TEDL.append(1)
    SL=list(range(0,len(DT2)))
    for i in range(0,round(len(DT2)*0.8)):
        r=np.random.randint(len(SL))
        del SL[r]
        k=DT2[r].split(',')
        ls=[]
        for j in range(0,len(k)):
            ls.append(float(k[j]))
        if len(ls)==((148+INX*2)*4):
            TRDT.append(ls)
            TRDL.append(0)
    for i in range(0,len(SL)):
        k=DT2[SL[i]].split(',')
        ls=[]
        for j in range(0,len(k)):
            ls.append(float(k[j]))
        if len(ls)==((148+INX*2)*4):
            TEDT.append(ls)
            TEDL.append(0)
    R1=len(TRDT)
    R2=len(TEDT)
    C1=len(TRDT[0])
    C2=len(TEDT[0])
    TRDT=np.array(TRDT)
    TEDT=np.array(TEDT)
    TRDL=np.array(TRDL)
    TEDL=np.array(TEDL)
    print(R1,C1,R2,C2)
    TRDT=TRDT.reshape((R1,1,C1,1))
    TEDT=TEDT.reshape((R2,1,C2,1))
    return TRDT,TEDT,TRDL,TEDL,R1,R2,C1,C2

import numpy as np
import copy
import sys
nd=sys.argv[1]
ST=int(sys.argv[2])
ND=[]
ND.append(nd)
for INX in range(ST,ST+1,10):
    for D in ND:
        for it in range(1,31):
            FN1=str(INX)+'/'+D+'PO'
            FN2=str(INX)+'/'+D+'NO'
            TRDT,TEDT,TRDL,TEDL,R1,R2,C1,C2=Prepare_Data(FN1,FN2,INX)
            input_shape = (1,C1)
            model = build_resnet(input_shape)
            model.summary()
            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            ne=100
            model.fit(TRDT,TRDL,epochs=1,batch_size=10,validation_data=(TEDT, TEDL),verbose=1)
            M1 = model.get_weights()
            res=model.evaluate(TEDT,TEDL,verbose=0)
            p=0
            OPT=['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam','ftrl']
            for epoch in range(ne):
                model.fit(TRDT,TRDL,epochs=1,batch_size=5,validation_data=(TEDT, TEDL),verbose=1)
                re=model.evaluate(TEDT,TEDL,verbose=0)
                print(INX,D,epoch,re,res)
                if (res[1]<re[1]):
                    p=0
                    print('Improvment',epoch,re[1])
                    res=copy.copy(re)
                    M1=model.get_weights()
                    F=open(str(INX)+'/'+D+'PRE'+str(it),'w')
                    PRE=model.predict(TEDT)
                    for i in range(0,len(TEDT)):
                        F.write(str(i)+','+str(PRE[i])+','+str(TEDL[i])+'\n')
                    PRE=model.predict(TRDT)
                    for i in range(0,len(TRDT)):
                        F.write(str(i)+','+str(PRE[i])+','+str(TRDL[i])+'\n')
                    F.close()
                elif p>3:
                    k=np.random.randint(len(OPT))
                    model.compile(loss='binary_crossentropy',optimizer=OPT[k],metrics=['accuracy'])
                    model.set_weights(M1)
                p=p+1