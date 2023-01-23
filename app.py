import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from wordcloud import WordCloud, STOPWORDS
import tensorflow.compat.v1 as tf
import fasttext

tf.disable_v2_behavior()




#"""Pull in Keywords which similarity will be based upon"""
keywords = pd.read_csv("Indeed_ED_raw_data_dump.csv")
keyword_list = keywords["Short_Description"]
keyword_list = [i for i in keyword_list if str(i) !='nan']



all_text = " "
for i in keyword_list:
    all_text+=i
with open('train.txt', 'w', encoding="utf-8") as f:
    f.write(all_text)


elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
ft = fasttext.train_supervised("train.txt")

#fit vectorizer on the keywords
#"""

def elmo_vectors(x):
  embeddings=elmo(x, signature="default", as_dict=True)["elmo"]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))



# Create a Vectorizer Object
vectorizer = CountVectorizer()
vectorizer.fit(keyword_list)

def similarity_score_scikit(job_description_a, job_description_b):
  jda = vectorizer.transform([job_description_a])
  A = jda.toarray()[0]
  jdb = vectorizer.transform([job_description_b])
  B = jdb.toarray()[0]
  similarity = float(np.dot(A,B)/(norm(A)*norm(B)))
  if np.isnan(similarity):
    return(0)
  else:
    return round(similarity*100,4)


def similarity_score_fasttext(job_description_a, job_description_b):
  A = ft.get_sentence_vector(job_description_a)
  B = ft.get_sentence_vector(job_description_b)
  print((cosine_similarity([A, B])))
  similarity = float(cosine_similarity([A, B])[0][1])
  if np.isnan(similarity):
    return(0)
  else:
    return round(similarity*100,4)

def similarity_score_elmo(job_description_a, job_description_b):
   A = elmo_vectors([job_description_a])[0]
   B = elmo_vectors([job_description_b])[0]
   similarity = float(cosine_similarity([A, B])[0][1])
   if np.isnan(similarity):
     return(0)
   else:
     return round(similarity*100,4)

def similarity_score(job_description_a, job_description_b, model):
  if model == 'Count vectorizer- scikit learn':
    return(similarity_score_scikit(job_description_a, job_description_b))
  elif model == 'ELMo':
    return(similarity_score_elmo(job_description_a, job_description_b))
  elif model == 'Fasttext':
    return(similarity_score_fasttext(job_description_a, job_description_b))


def plot_word_cloud(text_list):
  comment_words = ''
  stopwords = set(STOPWORDS)


  for val in text_list:
      
      # typecaste each val to string
      val = str(val)
  
      # split the value
      tokens = val.split()
      
      # Converts each token into lowercase
      for i in range(len(tokens)):
          tokens[i] = tokens[i].lower()
      
      comment_words += " ".join(tokens)+" "
  
  wordcloud = WordCloud(width = 800, height = 800,
                  background_color ='white',
                  stopwords = stopwords,
                  min_font_size = 10).generate(comment_words)
  return(wordcloud)

def main():
    
    # ===================== Set page config and background =======================
    # Main panel setup
    # Set website details
    st.set_page_config(page_title ="Job Design Similarity Scorer", 
                       page_icon=':desktop_computer:', 
                       layout='centered')
    """## Job Design Similarity Scorer"""

    with st.expander("About"):
        st.write("This App checks for the similarity between a user's job design and returns the score, There are 3 models here currently, CountVectorizer, FastText and ELMo")
   
    with st.expander("Settings"):
        model_option = st.selectbox('Kindly select preferred model',('Count vectorizer- scikit learn', 'ELMo','Fasttext'))
        # I used a slider to set-up an adjustable threshold
        demo = st.selectbox('Use Demo Texts',('No', 'Yes'))
    
    if demo == "Yes":
      demo_1 = "The ASPCA Communications Department is a passionate, collaborative team responsible for leveraging various communications channels to further the organization’s mission of preventing cruelty to dogs, cats, equines, and farm animals throughout the United States. As part of the Communications Department, the Social Media Assistant will work closely with the Social Media Manager to assist with influencer/celebrity outreach and engagement strategies, conduct social media listening and community management, and assist with social media content creation in support of the ASPCA’s programs aimed at improving the lives of animals. Responsibilities will include organizing data, maintaining databases with confidential celebrity/influencer information, building relationships with influencers, brainstorming compelling angles and writing pitches, tracking outreach lists, monitoring online conversation within multiple social media platforms and engaging as directed, and assisting with social media content creation."
      demo_2 = "Are you passionate about driving measurable impact to address wealth inequality through investing in education? Do you like developing and executing all aspects of large-scale projects and partnerships? Are you skilled at influencing others to engage in mission-driven programs? We’re looking for someone to: be part of building out our corporate philanthropy strategy focusing on education and skills provide program management and implementation support for our emerging education portfolio identify measurable outcomes, inform program planning, and execute the work to accelerate the mission utilize strong program and partnership management skills to manage an emerging portfolio of education programs and partnerships play key role in crafting program communications, marketing and thought leadership opportunities integrate value-add employee engagement opportunities into education and skills programming"
      demo_3 = "As a Knowledge Analyst (KA) within BCG's Social Impact Practice Area, you will work in a fast growing and entrepreneurial global team, providing expertise and insights for the ESG /DE&I topic to assist private sector clients determine which commercial levers and solutions will have the highest return on investment and implement on the ground those that drive the highest impact. ESG /DE&I is a driving force for BCG's ambition to become the most positively impactful company in the world. We believe that we can transform how business creates good competitive advantage, and that we can help profitably solve some of society’s most pressing problems together with our clients. You will collaborate with BCG case and proposal teams to deliver customized knowledge assets, analytics and expert advisory to our clients, in areas including DE&I, inclusive supply chains, reduction in pay gap initiatives, health equity, financial & digital inclusion and employee human rights. In addition, you will support your ESG DE&I topic team in developing intellectual property and managing content on internal BCG websites, ensuring availability of latest, high-quality materials. You will also support business development and go-to-market efforts, as opportunities arise, with research and analysis. We have a differentiated perspective in the market on Sustainability - a company’s economic, social, and environmental effects on the world - and can demonstrate that sustainability has also a positive impact on total shareholder return (TSR). We target delivering Socially Transformative Business transition s spanning a company’s products and services, business operations, and supply chains. We also aspire to imbed a sustainability lens in all of BCG’s work. You can find more about BCG's own sustainability ambition in BCG's latest sustainability report A Time to Lead: 2021 Annual Sustainability Report"
      demo_4 = "Youth To The People (YTTP) is looking for a passionate, strategic, and effective Social Impact Leader who will bring our brand’s vision and commitment to sustainability, DE&I, and social justice to life. The VP of Social Impact will be responsible for orchestrating the roadmap activation, roll out, and monitoring of our social impact strategy and initiatives. They will spearhead our nonprofits partnership program, be responsible for engaging and upskilling brand employees and stakeholders, and serve as an important voice to support our brand ethos and ensure YTTP continues to make a tangible impact in our community and on the planet."
      demo_5 = "The social media unit is part of the Brand Marketing team, which supports and collaborates with Strategic Communications and its partners, including but not limited to Executive Communications, Media Relations, Admissions Marketing, Experiential Marketing, Chancellor Communications, and Campus Marketing and Communications. Reporting to the Director of Social Media, the Social Media Copywriter works directly on the @ucla social media platforms to create, execute, measure, and advise on social media programs for UCLA Strategic Communications, in partnership with the creative team. The Social Media Copywriter is responsible for synthesizing news stories, marketing messages, and other brand storytelling into compelling text, as well as writing clear, high-impact copy for social media. This role will also team up with Issues Management and Communications teams to help craft language for statements and responses on social media. Primary responsibilities include crafting integrated messaging compatible with branding and marketing programs, collaborating with various creative teams across StratComm and Campus to shape their storytelling into engaging content for social media, and crafting messages for community/reputation management. This role will work on the day-to-day operations of UCLA's social media channels on Facebook, Twitter, Instagram/Stories/IGTV, LinkedIn, YouTube, TikTok and more and will producing content for these channels. Additional responsibilities: promoting UCLA to news media; monitoring for trends and engaging with followers/community; researching industry best practices for existing and emerging trends in social media; attending events (in-person when applicable) and providing coverage within and outside of normal business hours. The Social Media Copywriter collaborates with and serves as backup to the other Social Media Manager(s). Working hours may occasionally include weekends, evenings, and holidays and may require light travel."
      demo_6 = "At Freeman, we are all about the experience, and specifically, our People team is all about the Freeman Experience. We strive to deliver meaningful and impactful moments that matter to our people, create connection in meaningful ways, and we believe a critical component to success is creating positive social impact in the communities around us, which aligns with our values. Through Freeman Cares, we are committed to the well-being of our people, our industry, our communities, and our environment. We are currently seeking a Social Impact Specialist to reimagine the employee experience through the lens of social impact. Are you passionate about employee experience and connecting people more broadly with ways to create meaningful impact and empowering people to make the world a better place? The Social Impact Specialist will play an integral role in creating and implementing solutions and programs and creating strategic partnerships that our people care about to demonstrate tangibly that Freeman Cares. In this role, you will apply design thinking principles and analytical skills to help develop impact projects related to our social performance. You will have the opportunity to help shape executable projects and programs, execute the programs on the ground with our partners and community, as well as ensure development of robust communication and engagement strategies."
    else:
      demo_1 = ""
      demo_2 = ""
      demo_3 = ""
      demo_4 = ""
      demo_5 = ""
      demo_6 = ""

    with st.form(key = 'form1', clear_on_submit=False):
        Job_description1 = st.text_area("Job design and role crafting template data", value=demo_1)
        Job_description2 = st.text_area("First Job description",value=demo_2)
        Job_description3 = st.text_area("Second Job description",value=demo_3)
        Job_description4 = st.text_area("Third Job description",value=demo_4)
        Job_description5 = st.text_area("Fourth Job description",value=demo_5)
        Job_description6 = st.text_area("Fifth Job description",value=demo_6)
        submit_button = st.form_submit_button()

    if submit_button:
      job_description_list = [Job_design,Job_description1,Job_description2,Job_description3,Job_description4,Job_description4]
      corr = pd.DataFrame(index = ["job description {}".format(i) for i in range(1,7)])
      if model_option == "ELMo":
        st.warning("Warning !!! This will take some time - it requires patience")
      for i in range(1,7):
        corr["job description {}".format(i)] = [similarity_score(job_description_list[i-1],job_description_list[k-1], model_option) for k in range(1,7)]
        most_correlated = corr["job description 1"][1:].idxmax()
      st.success("I'm processing your request")
      st.write("The most correlated Job description is {}".format(most_correlated))
      with st.expander("See More Analysis"):
        fig = plt.figure(figsize=(20, 12))
        sns.set(font_scale=1.5)
        sns.heatmap(corr, cmap="Greens")
        plt.title('Heatmap of similarities between all the job descriptions')
        st.pyplot(fig)
        st.write("\n The Similarity score between your Job design and 1 is {}%".format(corr["job description 1"][1]))
        st.write("The Similarity score between your Job design and 2 is {}%".format(corr["job description 1"][2]))
        st.write("The Similarity score between your Job design and 3 is {}%".format(corr["job description 1"][3]))
        st.write("The Similarity score between your Job design and 4 is {}%".format(corr["job description 1"][4]))
        st.write("The Similarity score between your Job design and 5 is {}%\n".format(corr["job description 1"][5]))


        # plot the WordCloud image                      
        fig2 = plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(plot_word_cloud(job_description_list))
        plt.axis("off")
        plt.title("Word cloud of the job descriptions")
        plt.tight_layout(pad = 0)
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
