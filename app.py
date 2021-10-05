import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import datetime
from datetime import timedelta
from pandas import read_csv
from matplotlib import pyplot
from numpy import polyfit
from scipy import interpolate
from keras.models import load_model
from numpy import loadtxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta


@st.cache  ### significantly reduces the rerun time
def load_data():
    
    ## load PA_volume 
    volume = loadtxt('PA_volume.csv', delimiter=',')
    df = pd.read_csv("formulary.csv")
    pas = pd.read_csv('dim_pa.csv')
    claims = pd.read_csv('dim_claims.csv')
    bridge = pd.read_csv('bridge.csv')
    infile = open('log_reg.pcl','rb')
    log_reg_model = pickle.load(infile)
    infile.close()
    claims = claims.dropna()
    mega = pas.join(claims.set_index(pas.index))
    mega = mega.drop(columns=['dim_pa_id', 'dim_claim_id','pharmacy_claim_approved' ])
    mega = mega[['reject_code', 'correct_diagnosis', 'tried_and_failed', 'contraindication', 'pa_approved','drug',"bin"]]
    groups = [group.reset_index()[[ 'correct_diagnosis', 'tried_and_failed', 'contraindication', "bin",'reject_code','pa_approved']] for _, group in mega.groupby('drug')]
    return df, groups, log_reg_model, volume
    

def InFormulary(payer,drug,df):
    
    #Check for valid input
    if not(drug in df.columns):
        outstring = "Error! Drug is not valid."
    elif not( (df['Payer'] == float(payer)).any()):
        outstring = "Error! Payer is not valid."
    else:
        #Now check for if it is in formulary...
        sel = df.loc[df['Payer']==float(payer),['Acc/Code',drug]]
        
        #Code -1 is accepted
        acc = sel[sel['Acc/Code'] == -1].iat[0,1]
        
        #Code 70 means not in formulary
        nf = sel[sel['Acc/Code'] == 70].iat[0,1]
        
        #Code 75 means in formulary but not preferred
        np =  sel[sel['Acc/Code'] == 75].iat[0,1]
        
        outstring = ""
        
        if(acc > 0.5): outstring = "Drug is in formulary; no PA required unless plan limit is exceeded"
        
        if(np > 0.5): outstring = "Drug is on formulary, but not preferred; a PA will be required"
        
        if(nf > 0.5): outstring = "Drug is NOT on formulary; a PA will be required"

    return(outstring)
##
##
##
##
def convert(string):
    if string == 'Yes':
        return 1.0
    if string == 'No':
        return 0.0
def pa_approval(correct_diagnosis,tried_and_failed,contraindiction,binn,drug,reject_code,groups):
    if drug == "A":
        j = 0
    if drug == "B":
        j = 1
    if drug == "C":
        j = 2
    data = groups[j].to_numpy()
    count = 0
    approved = 0
    
    correct_diagnosis = convert(correct_diagnosis)
    tried_and_failed = convert(tried_and_failed)
    contraindiction = convert(contraindiction)
    for i in range(0,len(data)):

        if (data[i][:-1] == [float(correct_diagnosis),float(tried_and_failed),float(contraindiction),float(binn), float(reject_code)]).all():
            count += 1
            if data[i][-1] == 1:
                 approved += 1
    if count == 0:
        return "Your claim should not have been rejected"
    else:
        return  "Your PA approval rate is: "+ "{:.2f}".format(approved/count*100) +"%"
#print(claim_pred("A","417380"))
##
##
### In[ ]:
##
def ePAApprove(payer,
               drug,
               correct_diagnosis,
               tried_and_failed,
               contraindiction,
               not_in_formulary,
               limit_exceeded,log_reg_model ):
    correct_diagnosis = convert(correct_diagnosis)
    tried_and_failed = convert(tried_and_failed)
    contraindication = convert(contraindiction)
    not_in_formulary = convert(not_in_formulary)
    limit_exceeded = convert(limit_exceeded)
    #load pickle file
    

    #print(type(log_reg_model))
    
    #Put input parameters into a temporary data frame
    features = ['correct_diagnosis','tried_and_failed','contraindication',
                'not_in_formulary','limit_exceeded','Drug A','Drug B'
                ,417380, 417740, 417614]
    tmp = pd.DataFrame(columns=features)
    #print(tmp.head())

    tmp_list = [correct_diagnosis, tried_and_failed, contraindication,
                not_in_formulary, limit_exceeded]

    dl = [0,0]
    pl = [0,0,0]

    #We would need something more sophisticated for a larger number of drugs, but this will do for now.
    if(drug == 'A'): dl=[1,0]
    if(drug == 'B'): dl=[0,1]

    if(payer == 417380): pl = [1,0,0]
    if(payer == 417740): pl = [0,1,0]
    if(payer == 417614): pl = [0,0,1]
    #https://www.w3schools.com/python/python_lists_add.asp
    tmp_list.extend(dl)
    tmp_list.extend(pl)

 #   print(tmp_list)

    tmp = tmp.append(pd.DataFrame([tmp_list],columns=features))
  #  print(tmp.head())
    
    #Predict with unpickled (is that a word?) model

    prob = log_reg_model.predict_proba(tmp)[:,1]

    #Compare with cutoff (0.36), 
    approved = 'No' 
    if(prob>0.36): approved = 'Yes'

    return approved, prob

##def create_layout(df: pd.DataFrame,
##                  data_indicator: st.DeltaGenerator.DeltaGenerator, groups):
def create_layout(df,
                  data_indicator, groups, log_reg_model,  volume):

    data_indicator.subheader("✔️Data is loaded")
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Please select a page", ["Main page","Overview","Claim Prediction","ePA Prediction with Statistics","ePA Prediction with Machine Learning","ePA Volume Prediction" ])
    values = ['<select>','3','4']
    
    if app_mode == "Main page":
        
        st.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">SmartPA</p>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        .small-font {
            font-size:16px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="small-font">It is not always trivial to get your medication covered by your insurance!  <br />SmartPA is a web application that can provide data-driven advice to patients based on data analytics, predictive modeling. It can also help payers with budgeting by forecasting the volume of electronic Prior Authorizations (ePAs) with time series analysis. We have utilized anonymized data provided by CoverMyMeds as part of May-2021 Bootcamp organized by The Erdos Institute.</p>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        .small2-font {
            font-size:13px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="small2-font">Contributors: Yiming Gong, Luke Corwin, Prerna Kabtiyal,   Katherine Zhang (in that order)</p>', unsafe_allow_html=True)


    elif app_mode == "Overview":
        
        st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="big-font">Overview</p>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        .small-font {
            font-size:16px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="small-font">When a doctor prescribes a therapy to a patient, they send the prescription to a pharmacy. The pharmacy, when going to fill the prescription, runs a claim against the patient’s insurance (in this context known as a Pharmacy Benefit Manager or PBM), to see if their insurance will cover the therapy as prescribed (correct drug, dosage, etc.).</p>', unsafe_allow_html=True)
        st.markdown('<p class="small-font">However, payers may not cover particular medications or dosages, and may reject the claim. Claims can be rejected for a variety of reasons. For example, the dosage or quantity dispensed may not be covered, or the drug might not be on formulary. A formulary is a list of the preferred drugs a payer has. Typically, formularies are tiered, with some drugs being the cheapest, some being more costly but covered, and some requiring a prior authorization for example. </p>', unsafe_allow_html=True)
        st.markdown('<p class="small-font">When a claim is rejected, a “reject code” is provided which contextualizes the reason the claim was rejected. For our purposes, there are 3 rejection codes that we focus on: 70, 75, and 76. A code 70 implies that a drug is not covered by the plan and is not on formulary, and typically implies that another course of therapy should be pursued. A code 75 implies that a drug is on the formulary but does not have preferred status and requires a prior authorization (PA). A PA is a form that care providers submit on behalf of their patient making the case that the therapy as described is a critical course of treatment. A code 76 simply means that the drug is covered, but that the plan limitations have been exceeded, which means that the limitations on the number of fills for that medication has been met. We might expect there to be variation in the type of reject codes we see for certain drugs by the payer. </p>', unsafe_allow_html=True)
        st.markdown('<p class="small-font">If a claim is rejected, regardless of the reject code provided, a prior authorization can be started to prevent prescription abandonment and ensure a patient gets the therapy their provider thinks would work best for them. When a provider is filling out an ePA, they may frequently be asked to provide information about a patient’s diagnosis, lab values, contraindications (health-related reasons not to take certain medications), and if they have tried and failed other therapies. When reviewing prior authorizations, payers evaluate the information provided against their formulary and make a decision. That is to say, information contained on the PA and information contained in the original pharmacy claim can help us understand whether an ePA is likely to be approved or denied.</p>', unsafe_allow_html=True)
    elif app_mode == "Claim Prediction" :
        st.header("Claim Prediction")
        st.write("The Exploratory Data Analysis made it clear that each payer has different reasons for requiring a PA for each drug. The data provided specified this through a column of rejection codes which are identified as follows. 70.0- The drug is not on the provider\'s formulary, 75.0- The drug is on the payer\'s formulary but does not have preferred status, 76.0- The drug is on formulary but the patient has exceeded the allowed limit. We extracted the formulary of the three payers from over 100K medical records. Based on these formularies, we predict whether the medical claims would be rejected/a PA is required even before submissions.")
        
        payer = st.selectbox("Choose your Payer", ['417380','417614','417740','999001'], index=3)
        drug = st.selectbox("Choose your drug", ['A','B','C'], index=2)
        Claim_Prediction = InFormulary(payer,drug, df)
        st.write("The prediction of your medical claim: ")
        st.text(Claim_Prediction)
        
    elif app_mode == "ePA Prediction with Statistics" :
        st.title("ePA Prediction with Statistics")
        st.write("We can help you to predict the chance of your ePA getting approved! Our prediction model is empowered by statistics of over 100k ePA records from CoverMyMeds. Some random samples from the dataset are shown below:")
        st.write(groups[0].head())
        Payer = st.selectbox("Choose your Payer", ['417380','417614','417740','999001'], index=3)
        Drug = st.selectbox("Choose your drug", ['A','B','C'], index=2)
        Reject_code = st.selectbox("Rejection code", ['70','76','75'], index=2)
        Correct_diagnosis = st.selectbox("Correct_diagnosis", ['Yes','No'], index=1)
        Tried_and_failed = st.selectbox("Tried_and_failed", ['Yes','No'], index=1)
        Contraindiction = st.selectbox("Contraindiction", ['Yes','No'], index=1)
        stat_pred = pa_approval(Correct_diagnosis,Tried_and_failed,Contraindiction,Payer,Drug,Reject_code, groups)
        st.write("The prediction of your ePA:")
        st.text(stat_pred)
        
    elif app_mode == "ePA Prediction with Machine Learning" :
        st.title("ePA Prediction with Machine Learning")
        st.write("We tested the performance of machine learning techniques such as decision tree, random forest, logistic regression. The models were trained on the important features of ePA such as a patient’s diagnosis, lab values, contraindications (health-related reasons not to take certain medications), and if they have tried and failed other therapies." )

        st.write("Logistic Regression was found to be the best performing classification to predict the PA approval/rejection with high recall\(\~96%%\)\ and precision\(\~81%%\)\. A detailed comparison of the recall and precision can be found in our github repositary: https://github.com/lsky2061/code-2021-team-sapphire ")

        Payer = st.selectbox("Choose your Payer", ['417380','417614','417740','999001'], index=3)
        Drug = st.selectbox("Choose your drug", ['A','B','C'], index=2)
        Not_in_formulary = st.selectbox("Not_in_formulary", ['Yes','No'], index=1)
        Limit_exceeded = st.selectbox("Limit_exceeded", ['Yes','No'], index=1)
        Correct_diagnosis = st.selectbox("Correct_diagnosis", ['Yes','No'], index=1)
        Tried_and_failed = st.selectbox("Tried_and_failed", ['Yes','No'], index=1)
        Contraindiction = st.selectbox("Contraindiction", ['Yes','No'], index=1)
        approved, prob = ePAApprove(Payer,
               Drug,
               Correct_diagnosis,
               Tried_and_failed,
               Contraindiction,
               Not_in_formulary,
               Limit_exceeded,
                log_reg_model)
        st.write("Will my ePA get approved?")
        st.text(approved)
        st.write("Probability of approvale = ",prob[0])

    elif app_mode == "ePA Volume Prediction" :
        st.title("ePA Volume Prediction with Time Series Analysis")
        st.write(" We implemented a multilayer perceptron model, a LSTM model and the Facebook Prophet for the PA volume forecast. The three models were trained on a dataset that covers the PA volume from 2017-2020. Some preliminary observations of the dataset : the PA volume peaks around the start of the year, which is when many PAs expire and new ones will be resubmitted. The PA volume is at its highest during workdays, and its lowest during holiday seasons.")
        st.write("The yearly trend and weekly trend were both captured by all three models in different ways. We tested the accuracy of the three models by forecasting the volume in 2020 based on the data from 2017-2019. As a result, our self-trained LSTM model outperforms the Prophet by 3%% with a much faster runtime")
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=30)
        start_date = st.date_input('Start date', today)
        end_date = st.date_input('End date', tomorrow)
##        if start_date < end_date:
##            st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
##        else:
##            st.error('Error: End date must fall after start date.')
        LSTM = st.checkbox('PA Volume Predicted by LSTM Model')
        MLP = st.checkbox('PA Volume Predicted by Multilayer Perceptron Model')
        Prophet = st.checkbox('PA Volume Predicted by Prophet')
        plot_data = pd.DataFrame()
        max_length = 300
        if LSTM:
            wave_model = load_model('wave_model.h5')
            weekly_model = load_model('weekly_model.h5')
            #### Prepare data
            look_forward_wave = 35
            PA_volume = np.array(volume)
            workdays = []
            for i in range(len(PA_volume)):
                if i%7 <6 and i%7 > 0 :
                    workdays.append(PA_volume[i])
            workdays = np.array(workdays)
            workdays_short = []
            for i in range(0,(len(workdays) - 10),10):
                workdays_short.append(np.mean(workdays[i: i+10]))
            wave_idx = np.arange(len(workdays_short))
            wave_final_idx = np.arange(len(workdays))/len(workdays) * (len(workdays_short) - 1)

            tck = interpolate.splrep(wave_idx,workdays_short)
            wave_final = interpolate.splev(wave_final_idx, tck, der=0)

            look_forward_weekly = int(len(PA_volume) * look_forward_wave/len(workdays_short))
            station = PA_volume.copy()
            j = 0
            for i in range(len(station)):
                if i%7 <6 and i%7 > 0 :
                    station[i] = PA_volume[i] - wave_final[j] + 600
                    j += 1
                    
            station_scaler = MinMaxScaler(feature_range=(0, 1))
            station = np.array(station).reshape(-1,1)
            station = station_scaler.fit_transform(station)

            #### Get the wave ground truth
            #### Get prediction of weekly volume

            weekly_end = np.expand_dims(station[-look_forward_weekly:,0 ], axis = 0)
            weeklyPredict = weekly_model.predict(np.reshape(weekly_end, (weekly_end.shape[0], 1, weekly_end.shape[1])))
            # invert predictions
            weeklyPredict = np.squeeze(station_scaler.inverse_transform(weeklyPredict))

            ##### Predict the wave
            station_scaler = MinMaxScaler(feature_range=(0, 1))
            wave_scaler = MinMaxScaler(feature_range=(0, 1))
            workdays_short = np.array(workdays_short).reshape(-1,1)
            workdays_short = wave_scaler.fit_transform(workdays_short)
            wave_end = np.expand_dims(workdays_short[-look_forward_wave:,0 ], axis = 0)
            wavePredict = wave_model.predict(np.reshape(wave_end, (wave_end.shape[0], 1, wave_end.shape[1])))
            wavePredict = np.squeeze(wave_scaler.inverse_transform(wavePredict))
            wavePredict_idx = np.arange(len(wavePredict))
            wavePredict_long_idx = np.arange(10 * len(wavePredict))/10
            tck = interpolate.splrep(wavePredict_idx,wavePredict)
            wavePredict_long = interpolate.splev(wavePredict_long_idx, tck, der=0)
            weeklyPredict = weeklyPredict[:max_length]
            final = weeklyPredict.copy()
            j = 0
            for i in range(len(weeklyPredict)-1):
                if i%7 != 3 and i%7 != 4 :
                    final[i] = weeklyPredict[i] + wavePredict_long[j] - 600
                    j += 1

            plot_data['LSTM'] = final
##        
        if MLP:
            #### Generate a function for this method 
            #### calculate the PA volume
            PA_volume = volume
            #### Create the date stamp for PA volume
            Start_date = datetime.date(2018, 12, 31)
            datestamp = []
            for i in range(len(PA_volume)):
                datestamp.append(Start_date)
                Start_date = Start_date + timedelta(days = 1)
            PA = {'time':datestamp,
                    'data':PA_volume}
            df = pd.DataFrame(PA) 
            Ntrain = 340  ### the length of train chunks
            Nlabel = 340  ### the length of prediction
            dates = np.asarray(datestamp)
            he = pd.DataFrame({'data': PA_volume}, index=dates)
            period=50
            decomposition = seasonal_decompose(he, model='additive', period=period)
            df_trend = pd.DataFrame({'data': PA_volume, 'trend': decomposition.trend}, index=dates)
            df_trend = df_trend.dropna()
            #### Stationarize the time series
            stationary = []
            for i in range(len(df_trend)):
                if i%7 != 2 and i%7 != 3:
                    stationary.append(df_trend.iloc[i,0] - df_trend.iloc[i,1] + 400)
                else:
            #         stationary.append(df_trend.iloc[i,0])
                    stationary.append(df_trend.iloc[i,0] - df_trend.iloc[i,1] + 400)

            def series_to_supervised(data, n_in, n_out):
                train, label = [],[]
                for i in range(0, len(data) - n_out - n_in):
                    train.append(data[i:i + n_in])
                    label.append(data[i + n_in: i + n_in + n_out])
                return np.array(train), np.array(label)
            trend = df_trend['trend'].tolist()
            #### Create training sets
            train_trend, label_trend = series_to_supervised(trend,Ntrain, Nlabel )
            #### Create MLP model
            model = keras.models.load_model('MLP_stationary')
            model_trend = keras.models.load_model('trend_model')

            # model_trend = Sequential()
            # model_trend.add(Dense(100, activation='relu', input_dim=Ntrain))
            # model_trend.add(Dense(Nlabel))
            # model_trend.compile(optimizer='adam', loss='mse')
            # model_trend.fit(train_trend, label_trend, epochs=2000, verbose=0)
            #### get the last Ntrain observations
            sample_trend = np.array(trend[-Ntrain:])
            input_trend = sample_trend.reshape((1, Ntrain))
            ##### make prediction on the future trend
            yhat_trend = np.squeeze(model_trend.predict(input_trend, verbose=0))
            trend_predict = list(trend) + list(yhat_trend)
            ##### Prepare the training chunks for stationary series prediction
            train_data, label_data = series_to_supervised(stationary,Ntrain, Nlabel )
            # model = Sequential()
            # model.add(Dense(100, activation='relu', input_dim=Ntrain))
            # model.add(Dense(Nlabel))
            # model.compile(optimizer='adam', loss='mse')
            # model.fit(train_data, label_data, epochs=2000, verbose=0)
            # x_input = array([70, 80, 90])
            sample = np.array(stationary[-Nlabel-Ntrain:-Nlabel])
            sample_label = np.array(stationary[-Nlabel:])
            x_input = sample.reshape((1, Ntrain))
            yhat = model.predict(x_input, verbose=0)
            train_and_predict = stationary.copy()
            train_and_predict += list(np.squeeze(yhat))
            #### Add back the prediction of trend
            for i in range(0, len(train_and_predict)):
                if i%7 != 2 and i%7 != 3:
                    train_and_predict[i] = train_and_predict[i] + trend_predict[i] - 400
                else:
            #         pass
                    train_and_predict[i] = train_and_predict[i] + trend_predict[i] - 400

            future_prediction = train_and_predict[- Nlabel + int(period/2) + 4:] ### take the part omitted by the trend estimation into account
            
            plot_data['MLP'] =  future_prediction[:max_length]

        if Prophet:
            from prophet import Prophet
            period = 340
            Start_date = datetime.date(2018, 1, 1)
            PA_volume = volume
            datestamp = []
            for i in range(len(PA_volume)):
                datestamp.append(Start_date)
                Start_date = Start_date + timedelta(days = 1)
            PA = {'ds':datestamp,
                    'y':PA_volume}
            PA = pd.DataFrame(PA)
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(PA)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            plot_data['Prophet'] =  forecast['yhat'].to_list()[-period: -period + max_length]

        date_begin = datetime.date(2018, 1, 1)
        client_start_idx = int((start_date - date_begin)//timedelta(days=1))
        client_end_idx = int((end_date - date_begin)//timedelta(days=1))
        datestamp = []
        for i in range(client_end_idx - client_start_idx):
            datestamp.append(start_date)
            start_date = start_date + timedelta(days = 1)
        fig, ax = plt.subplots()
        color_list = ['r','g', 'y']
        
        if client_start_idx < 3 * 356 and client_end_idx > 3 * 356:
            
            
            ax.plot(datestamp[:(len(PA_volume) - client_start_idx)], PA_volume[client_start_idx:], color='b', label = 'database')

            i = 0
            for col in plot_data.columns:
                ax.plot(datestamp[(len(PA_volume) - client_start_idx):], plot_data[col].tolist()[:client_end_idx - len(PA_volume)], color=color_list[i], label = col)
##                ax.plot(, plot_data[col].tolist()[:client_end_idx - len(PA_volume)], color=color_list[i], label = col)

                i += 1
        elif client_start_idx < 3 * 356 and client_end_idx < 3 * 356:
            
            ax.plot(datestamp, PA_volume[client_start_idx:client_end_idx], color='b', label = 'database')
##            ax.plot(, PA_volume[client_start_idx:client_end_idx], color='b', label = 'database')

        elif client_start_idx > 3 * 356 and client_end_idx > 3 * 356:
            i = 0
            for col in plot_data.columns:
##                ax.plot(np.arange(0, client_end_idx - client_start_idx ), plot_data[col].tolist()[client_start_idx - len(PA_volume):client_end_idx - len(PA_volume)], color=color_list[i], label = col)
                ax.plot(datestamp, plot_data[col].tolist()[client_start_idx - len(PA_volume):client_end_idx - len(PA_volume)], color=color_list[i], label = col)

                i += 1
##        if 
##        Nbins = 5
##        
##        for i in range(len(date_axis_list):
##            t = date_axis_list[i]
##            t.strftime('%m/%d/%Y')
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        import matplotlib.dates as mdates
        majorLocator   = mdates.DayLocator(interval = int((client_end_idx - client_start_idx)/5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        minorLocator   = MultipleLocator(1)

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
                       
        leg = ax.legend()
        st.pyplot(fig)
##        st.line_chart(plot_data)


def main():
    data_indicator = st.sidebar.subheader("⭕️ Loading data")
    df, groups, log_reg_model, volume= load_data()
    create_layout(df,data_indicator,groups,log_reg_model, volume)


if __name__ == "__main__":
    
    main()
