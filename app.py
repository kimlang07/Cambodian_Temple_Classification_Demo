import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# Hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Cambodia Historical Temple Recognition",
    page_icon="AngkorWat.png",
    initial_sidebar_state='auto'
)

# Hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

# Load the model function
# @st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('VGG16_model_75.h5')
    return model

# Load the model
with st.spinner('Model is being loaded..'):
    model = load_model()

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    size = (150, 150)  # Changed to match the model's expected input size
    image = Image.open(uploaded_file)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0  # Rescale the image
    return img_reshape

# Define the function to get the class label from prediction
def prediction_cls(prediction, class_names):
    return class_names[np.argmax(prediction)]

# Sidebar contents
with st.sidebar:
    st.markdown("""
    <h1 style='text-align: center;'>Cambodia Historical Temple Recognition</h1>""", unsafe_allow_html=True)
    st.image('AngkorWat.png')
    # st.title("Cambodia Historical Temple Recognition")
    # st.subheader("Accurately classify the temples in Cambodia. This helps a user to get to know temples in Cambodia clearly.")

# Justified subheader using custom HTML and CSS
    st.markdown("""
    <h3 style='text-align: justify;'>
        Accurately classify the temples in Cambodia. This helps a user to get to know temples in Cambodia clearly.
    </h3>""", unsafe_allow_html=True)

# Main content
st.markdown("""
    <h2 style='text-align: center;'>Cambodia Historical Temple Recognition</h2>""", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    # st.text("Please upload an image of a temple.")
    st.markdown("""
    <h4 style='text-align: center;'>សូមដាក់រូបប្រាសាទនៅទីនេះ។</h4>""", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='text-align: center;'>Please upload an image of a temple.</h4>""", unsafe_allow_html=True)
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    processed_image = preprocess_image(file)
    predictions = model.predict(processed_image)

    class_names = ['Angkor_Wat', 'Bayon', 'Koh_Ker', 'Prasat Sambor Prei Kuk', 'Preah_Vihear']
    predicted_class = prediction_cls(predictions, class_names)

    # st.sidebar.error(f"Accuracy: {accuracy:.2f} %")

    result_message = f"Detected Temple: {predicted_class}"

#begin
    # Display the accuracy percentages for each class
    st.markdown("<h4 style='text-align: center;'>Prediction Probabilities:</h4>", unsafe_allow_html=True)
    
    # Get the softmax values and display them with percentages
    for i, class_name in enumerate(class_names):
        probability = predictions[0][i] * 100  # Convert to percentage
        st.markdown(f"<h5 style='text-align: center;'>{class_name}: {probability:.2f}%</h5>", unsafe_allow_html=True)
# end

    if predicted_class == 'Angkor_Wat':
        # st.balloons()
        st.sidebar.success(result_message)
        st.markdown("## ប្រាសាទអង្គរវត្ត - Angkor Wat")

        st.markdown("#### ពត៌មានខ្លះៗនៃប្រាសាទអង្គរវត្ត៖")
        html_text_AKW_KH = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            អង្គរជាទីតាំងបុរាណវត្ថុដ៏សំខាន់បំផុតមួយនៅអាស៊ីអាគ្នេយ៍។ លាតសន្ធឹងលើផ្ទៃដីប្រមាណ ៤០០ គីឡូម៉ែត្រក្រឡា រួមទាំងតំបន់ព្រៃឈើ ឧទ្យានបុរាណវិទ្យាអង្គរផ្ទុកនូវអដ្ឋិធាតុដ៏អស្ចារ្យនៃរាជធានីផ្សេងៗគ្នានៃអាណាចក្រខ្មែរចាប់ពីសតវត្សទី ៩ ដល់សតវត្សទី ១៥ ។ ប្រាសាទទាំងនោះរួមមានប្រាសាទអង្គរវត្តដ៏ល្បីល្បាញ និងនៅអង្គរធំ ប្រាសាទបាយ័ន ជាមួយនឹងការតុបតែងចម្លាក់រាប់មិនអស់។ អង្គការយូណេស្កូបានបង្កើតកម្មវិធីទូលំទូលាយមួយដើម្បីការពារទីតាំងនិមិត្តសញ្ញានេះ និងតំបន់ជុំវិញរបស់វា។

            អង្គរ​នៅ​ខេត្ត​សៀម​រាប​ភាគ​ខាងជើង​នៃ​ប្រទេស​កម្ពុជា​គឺ​ជា​រមណីយដ្ឋាន​បុរាណ​វិទ្យា​ដ៏​សំខាន់​បំផុត​មួយ​នៃ​អាស៊ីអាគ្នេយ៍។ វាលាតសន្ធឹងជាង 400 គីឡូម៉ែត្រការ៉េ និងមានប្រាសាទជាច្រើន រចនាសម្ព័ន្ធធារាសាស្ត្រ (អាង ទំនប់ អាងស្តុកទឹក ប្រឡាយ) ក៏ដូចជាផ្លូវទំនាក់ទំនង។ អស់រយៈពេលជាច្រើនសតវត្សមកហើយ អង្គរគឺជាមជ្ឈមណ្ឌលនៃព្រះរាជាណាចក្រខ្មែរ។ ជាមួយនឹងវិមានដ៏គួរឱ្យចាប់អារម្មណ៍ ផែនការទីក្រុងបុរាណផ្សេងៗគ្នា និងអាងស្តុកទឹកដ៏ធំ ទីតាំងនេះគឺជាការប្រមូលផ្តុំពិសេសនៃលក្ខណៈពិសេសដែលបញ្ជាក់ពីអារ្យធម៌ពិសេសមួយ។ ប្រាសាទដូចជាប្រាសាទអង្គរវត្ត បាយ័ន ព្រះខ័ន និងប្រាសាទតាព្រហ្ម ដែលជាគំរូនៃស្ថាបត្យកម្មខ្មែរត្រូវបានផ្សារភ្ជាប់យ៉ាងជិតស្និទ្ធទៅនឹងបរិបទភូមិសាស្រ្តរបស់ពួកគេ ក៏ដូចជាត្រូវបានបង្កប់ដោយសារៈសំខាន់ជានិមិត្តរូប។ ស្ថាបត្យកម្ម និងប្លង់នៃរាជធានីជាប់ៗគ្នា ជាសក្ខីភាពនៃសណ្តាប់ធ្នាប់សង្គម និងចំណាត់ថ្នាក់ខ្ពស់នៅក្នុងអាណាចក្រខ្មែរ។ ដូច្នេះ អង្គរគឺជាទីតាំងដ៏សំខាន់មួយដែលបង្ហាញពីតម្លៃវប្បធម៌ សាសនា និងនិមិត្តសញ្ញា ព្រមទាំងមានអត្ថន័យស្ថាបត្យកម្ម បុរាណវិទ្យា និងសិល្បៈខ្ពស់។

            ឧទ្យាននេះមានប្រជាជនរស់នៅហើយភូមិជាច្រើនដែលបុព្វបុរសមានអាយុកាលតាំងពីសម័យអង្គរត្រូវបានរាយប៉ាយពសពេញឧទ្យាន។ ប្រជាជន​ប្រកប​របរ​កសិកម្ម និង​ពិសេស​ជាង​នេះ​ទៀត​គឺ​ការ​ធ្វើ​ស្រែ។

            អង្គរគឺជាកន្លែងបុរាណវិទ្យាដ៏ធំបំផុតមួយដែលកំពុងដំណើរការនៅក្នុងពិភពលោក។ ទេសចរណ៍តំណាងឱ្យសក្តានុពលសេដ្ឋកិច្ចដ៏ធំសម្បើម ប៉ុន្តែវាក៏អាចបង្កើតការបំផ្លិចបំផ្លាញដែលមិនអាចជួសជុលបាននៃរូបី ក៏ដូចជាបេតិកភណ្ឌវប្បធម៌អរូបីផងដែរ។ គម្រោងស្រាវជ្រាវជាច្រើនត្រូវបានអនុវត្ត ចាប់តាំងពីកម្មវិធីការពារអន្តរជាតិត្រូវបានដាក់ឱ្យដំណើរការជាលើកដំបូងនៅក្នុងឆ្នាំ 1993។ គោលបំណងវិទ្យាសាស្ត្រនៃការស្រាវជ្រាវ (ឧ. ការសិក្សាផ្នែកនរវិទ្យាលើលក្ខខណ្ឌសេដ្ឋកិច្ចសង្គម) បណ្តាលឱ្យមានចំណេះដឹង និងការយល់ដឹងកាន់តែប្រសើរឡើងអំពីប្រវត្តិនៃគេហទំព័រ និងរបស់វា។ អ្នកស្រុកដែលបង្កើតជាកេរដំណែលពិសេសដ៏សម្បូរបែបនៃបេតិកភណ្ឌអរូបី។ គោលបំណងគឺដើម្បីភ្ជាប់ "វប្បធម៌អរូបី" ទៅនឹងការលើកកំពស់បូជនីយដ្ឋាន ដើម្បីដាស់តឿនប្រជាជនក្នុងតំបន់អំពីសារៈសំខាន់ និងភាពចាំបាច់នៃការការពារ និងការអភិរក្សរបស់វា និងជួយក្នុងការអភិវឌ្ឍន៍រមណីយដ្ឋាន ព្រោះអង្គរជាតំបន់បេតិកភណ្ឌរស់ជាតិខ្មែរ។ ប្រជាជនជាទូទៅ ប៉ុន្តែជាពិសេសប្រជាជនក្នុងតំបន់ ត្រូវបានគេស្គាល់ថាជាអ្នកអភិរក្សជាពិសេសទាក់ទងនឹងទំនៀមទម្លាប់ដូនតា និងជាកន្លែងដែលពួកគេប្រកាន់ខ្ជាប់នូវទំនៀមទម្លាប់វប្បធម៌បុរាណជាច្រើនដែលបានបាត់ទៅកន្លែងផ្សេង។ អ្នក​ស្រុក​គោរព​បូជា​អាទិទេព​ក្នុង​ព្រះវិហារ ហើយ​រៀបចំ​ពិធី និង​ពិធី​គោរព​បូជា ដោយ​មាន​ការ​បួងសួង ភ្លេង​ប្រពៃណី និង​របាំ។ ម្យ៉ាងទៀត ឧទ្យានបុរាណវិទ្យាអង្គរសម្បូរទៅដោយរុក្ខជាតិឱសថ ដែលប្រជាជនក្នុងតំបន់ប្រើប្រាស់សម្រាប់ព្យាបាលជំងឺ។ រុក្ខជាតិ​ត្រូវ​បាន​រៀបចំ​រួច​ហើយ​នាំ​យក​ទៅ​កាន់​កន្លែង​ប្រាសាទ​ផ្សេង​ៗ​គ្នា​ដើម្បី​ទទួល​ពរ​ពី​ព្រះ។ ប្រាសាទ​ព្រះខ័ន​ត្រូវ​បាន​គេ​ចាត់​ទុក​ថា​ជា​សាកលវិទ្យាល័យ​វេជ្ជសាស្ត្រ និង​នាគព័ន្ធ​ជា​មន្ទីរពេទ្យ​បុរាណ។ ទិដ្ឋភាពនៃបេតិកភណ្ឌអរូបីទាំងនេះត្រូវបានពង្រឹងបន្ថែមដោយការអនុវត្តវាយនភណ្ឌ និងកន្ត្រកប្រពៃណី និងការផលិតស្ករត្នោត ដែលលទ្ធផលទាំងអស់នេះជាផលិតផលដែលត្រូវបានលក់នៅលើទីផ្សារក្នុងស្រុក និងដល់ភ្ញៀវទេសចរ ដូច្នេះហើយបានរួមចំណែកដល់ការអភិវឌ្ឍប្រកបដោយចីរភាព និងជីវភាពរស់នៅរបស់ប្រជាជនរស់នៅ។ ក្នុង និងជុំវិញតំបន់បេតិកភណ្ឌពិភពលោក។
            <br><br>
            <a href="https://whc.unesco.org/en/list/668/" style="color: #d30000; text-decoration: none;" target="_blank">ប្រភព៖ UNESCO</a>
            </div>
            """
        st.markdown(html_text_AKW_KH, unsafe_allow_html=True)

        st.markdown("#### Some information of Angkor Wat Temple:")
        html_text_AKW_EN = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            Angkor is one of the most important archaeological sites in South-East Asia. Stretching over some 400 km2, including forested area, Angkor Archaeological Park contains the magnificent remains of the different capitals of the Khmer Empire, from the 9th to the 15th century. They include the famous Temple of Angkor Wat and, at Angkor Thom, the Bayon Temple with its countless sculptural decorations. UNESCO has set up a wide-ranging programme to safeguard this symbolic site and its surroundings.
            
            Angkor, in Cambodia’s northern province of Siem Reap, is one of the most important archaeological sites of Southeast Asia. It extends over approximately 400 square kilometres and consists of scores of temples, hydraulic structures (basins, dykes, reservoirs, canals) as well as communication routes. For several centuries Angkor, was the centre of the Khmer Kingdom. With impressive monuments, several different ancient urban plans and large water reservoirs, the site is a unique concentration of features testifying to an exceptional civilization. Temples such as Angkor Wat, the Bayon, Preah Khan and Ta Prohm, exemplars of Khmer architecture, are closely linked to their geographical context as well as being imbued with symbolic significance. The architecture and layout of the successive capitals bear witness to a high level of social order and ranking within the Khmer Empire. Angkor is therefore a major site exemplifying cultural, religious and symbolic values, as well as containing high architectural, archaeological and artistic significance.

            The park is inhabited, and many villages, some of whom the ancestors are dating back to the Angkor period are scattered throughout the park. The population practices agriculture and more specifically rice cultivation.

            Angkor is one of the largest archaeological sites in operation in the world. Tourism represents an enormous economic potential but it can also generate irreparable destructions of the tangible as well as intangible cultural heritage. Many research projects have been undertaken, since the international safeguarding program was first launched in 1993.The scientific objectives of the research (e.g. anthropological studies on socio-economic conditions) result in a better knowledge and understanding of the history of the site, and its inhabitants that constitute a rich exceptional legacy of the intangible heritage. The purpose is to associate the “intangible culture” to the enhancement of the monuments in order to sensitize the local population to the importance and necessity of its protection and preservation and assist in the development of the site as Angkor is a living heritage site where Khmer people in general, but especially the local population, are known to be particularly conservative with respect to ancestral traditions and where they adhere to a great number of archaic cultural practices that have disappeared elsewhere. The inhabitants venerate the temple deities and organize ceremonies and rituals in their honor, involving prayers, traditional music and dance. Moreover, the Angkor Archaeological Park is very rich in medicinal plants, used by the local population for treatment of diseases. The plants are prepared and then brought to different temple sites for blessing by the gods. The Preah Khan temple is considered to have been a university of medicine and the NeakPoan an ancient hospital. These aspects of intangible heritage are further enriched by the traditional textile and basket weaving practices and palm sugar production, which all result in products that are being sold on local markets and to the tourists, thus contributing to the sustainable development and livelihood of the population living in and around the World Heritage site.
            <br><br>
            <a href="https://whc.unesco.org/en/list/668/" style="color: #d30000; text-decoration: none;" target="_blank">Source: UNESCO</a>
            </div>
            """
        st.markdown(html_text_AKW_EN, unsafe_allow_html=True)

    elif predicted_class == 'Bayon':
        st.sidebar.warning(result_message)
        st.markdown("## ប្រាសាទបាយ័ន - Bayon Temple")
        st.markdown("#### ពត៌មានខ្លះៗនៃប្រាសាទបាយ័ន៖")
        html_text_BY_KH = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            នៅចំកណ្តាលអង្គរធំគឺប្រាសាទបាយ័នសតវត្សទី 12 ដែលជាប្រាសាទដ៏គួរឱ្យស្ញប់ស្ញែង ប្រសិនបើរំជួលចិត្តបន្តិច គឺជាប្រាសាទរដ្ឋរបស់ព្រះបាទជ័យវរ្ម័នទី៧។ វាបង្ហាញពីភាពប៉ិនប្រសប់ប្រកបដោយភាពច្នៃប្រឌិត និងគុណបំណាច់របស់ព្រះមហាក្សត្រដ៏ល្បីបំផុតរបស់កម្ពុជា។ ប៉មហ្គោធិកចំនួន 54 របស់វាត្រូវបានតុបតែងជាមួយនឹងមុខញញឹមដ៏អស្ចារ្យចំនួន 216 របស់ Avalokiteshvara ហើយវាត្រូវបានតុបតែងដោយចម្លាក់លៀនស្រាល 1.2 គីឡូម៉ែត្រដែលមានរូបចម្លាក់ជាង 11,000 ។
            អាថ៌កំបាំង​ជុំវិញ​ឈ្មោះ​ប្រាសាទ​រួម​ចំណែក​ដល់​ប៉ម​មុខ​ដ៏​អស្ចារ្យ​ដែល​កំណត់​ស្ថាបត្យកម្ម​របស់​ប្រាសាទ។ ចាប់តាំងពីការរកឃើញឡើងវិញរបស់បាយ័នដោយអ្នកប្រាជ្ញ និងអ្នករុករកជនជាតិបារាំងក្នុងសតវត្សទី 19 អត្តសញ្ញាណនៃរូបញញឹមដែលស្វាគមន៍អ្នកទស្សនាប្រាសាទ និងទីក្រុងអង្គរធំត្រូវបានពិភាក្សា។ តើ​ពួកគេ​ពណ៌នា​អំពី​ពុទ្ធសាសនា ឬ​អាទិទេព​ហិណ្ឌូ ឬ​តើ​ពួកគេ​ពណ៌នា​ព្រះបាទ​ជ័យវរ្ម័នទី ៧ ញញឹម​លើ​អាណាចក្រ​របស់​ព្រះអង្គ​? សិលាចារឹកពីព្រះវិហារបរិសុទ្ធរួមជាមួយឈ្មោះរបស់វាអាចជួយក្នុងការយល់ដឹងរបស់យើងអំពីអត្តសញ្ញាណនៃតួរលេខមុខបួន។ ទោះបីជាយ៉ាងណាក៏ដោយ វាត្រូវបានសន្មតដោយអ្នកប្រាជ្ញថា សិលាចារឹកបែបនេះត្រូវបានបំផ្លាញដោយអ្នកគ្រប់គ្រងបន្តបន្ទាប់ ដែលបានកាន់កាប់អង្គរធំបន្ទាប់ពីការសោយទីវង្គត់របស់ព្រះបាទជ័យវរ្ម័នទី៧។
            <br><br>
            <a href="https://smarthistory.org/bayon-temple-angkor-thom/" style="color: #d30000; text-decoration: none;" target="_blank">ប្រភព៖ Smart History</a>
            </div>
            """
        st.markdown(html_text_BY_KH, unsafe_allow_html=True)

        st.markdown("#### Some information of Bayon Temple:")
        html_text_BY_EN = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            At the heart of Angkor Thom is the 12th-century Bayon, the mesmerising, if slightly mind-bending, state temple of Jayavarman VII. It epitomises the creative genius and inflated ego of Cambodia’s most celebrated king. Its 54 Gothic towers are decorated with 216 gargantuan smiling faces of Avalokiteshvara, and it is adorned with 1.2km of extraordinary bas-reliefs incorporating more than 11,000 figures.
            The mystery over the temple’s name contributes to the enigmatic face-towers that define the temple’s architecture. Since the Bayon’s rediscovery by French scholars and explorers in the 19th century, the identity of the smiling figures that greet visitors to the temple and to the city of Angkor Thom has been debated. Do they depict Buddhist or Hindu deities, or do they depict King Jayavarman VII smiling over his empire? Inscriptions from the temple along with its name could aid in our understanding of the identity of the four-faced figures; however, it has been presumed by scholars that such inscriptions were destroyed by successive rulers who took over Angkor Thom after the death of Jayavarman VII. 
            <br><br>
            <a href="https://smarthistory.org/bayon-temple-angkor-thom/" style="color: #d30000; text-decoration: none;" target="_blank">Source: Smart History</a>
            </div>
            """
        st.markdown(html_text_BY_EN, unsafe_allow_html=True)
    elif predicted_class == 'Koh_Ker':
        st.sidebar.warning(result_message)
        st.markdown("## ប្រាសាទកោះកេរ - Koh Ker")
        st.markdown("#### ពត៌មានខ្លះៗនៃប្រាសាទកោះកេរ៖")
        html_text_KK_KH = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            ទីតាំងបុរាណវត្ថុនៃកោះកេរ គឺជាក្រុមទីប្រជុំជនដ៏ពិសិដ្ឋនៃប្រាសាទ និងទីសក្ការៈជាច្រើន រួមទាំងចម្លាក់ សិលាចារឹក គំនូរជញ្ជាំង និងសំណល់បុរាណវិទ្យា។ សាងសង់ក្នុងរយៈពេលម្ភៃបីឆ្នាំ វាជារាជធានីមួយក្នុងចំនោមរាជធានីរបស់អាណាចក្រខ្មែរដែលជាគូប្រជែងពីរ មួយទៀតគឺអង្គរ ហើយជារាជធានីតែមួយគត់ពីឆ្នាំ ៩២៨ ដល់ ៩៤៤ នៃគ.ស។ បង្កើតឡើងដោយព្រះបាទជ័យវរ្ម័នទី៤ ទីក្រុងដ៏ពិសិដ្ឋរបស់ទ្រង់ត្រូវបានគេជឿថាត្រូវបានដាក់ចេញដោយផ្អែកលើគោលគំនិតសាសនាឥណ្ឌាបុរាណនៃសកលលោក។ ទីក្រុងថ្មីបានបង្ហាញពីការធ្វើផែនការទីក្រុងមិនធម្មតា ការបញ្ចេញមតិសិល្បៈ និងបច្ចេកវិទ្យាសំណង់ ជាពិសេសការប្រើប្រាស់ដុំថ្ម monolithic ដ៏ធំ។
            កោះកេរ៖ រមណីយដ្ឋានបុរាណវត្ថុបុរាណ លីងបុរ ឬ ចក ហ្គាហ្គី ជារាជធានីនៃអាណាចក្រខ្មែរចន្លោះឆ្នាំ ៩២១ ដល់ ៩៤៤ គ.ស.។ មួយផ្នែកលាក់ខ្លួននៅក្នុងព្រៃស្លឹកធំទូលាយរវាងជួរភ្នំដងរែក និងភ្នំគូលែន នៅលើភ្នំដែលមានជម្រាលយ៉ាងទន់ភ្លន់ចម្ងាយប្រហែលប៉ែតសិបគីឡូម៉ែត្រភាគឦសាននៃអង្គរ រមណីយដ្ឋានបុរាណវិទ្យាមានប្រាសាទ និងទីជម្រកជាច្រើនដែលមានរូបចម្លាក់ សិលាចារឹក និងគំនូរជញ្ជាំង សំណល់បុរាណវត្ថុ និងធារាសាស្ត្រ។ រចនាសម្ព័ន្ធ។ ត្រូវបានបង្កើតឡើងដោយព្រះបាទជ័យវរ្ម័នទី៤ ក្នុងឆ្នាំ ៩២១ នៃគ.ស. កោះកេរ គឺជារាជធានីមួយក្នុងចំណោមរាជធានីគូប្រជែងពីរនៃអាណាចក្រខ្មែរ ដែលរួមរស់រវាងឆ្នាំ ៩២១ និង ៩២៨ នៃគ.ស មួយទៀតជាអង្គរ និងរាជធានីតែមួយគត់រហូតដល់ឆ្នាំ ៩៤៤ នៃគ.ស ក្រោយមកមជ្ឈមណ្ឌលនយោបាយរបស់អាណាចក្រ។ ត្រឡប់ទៅអង្គរវិញ។ សាងសង់ក្នុងដំណាក់កាលតែមួយក្នុងរយៈពេលម្ភៃបីឆ្នាំ ទីក្រុងដ៏ពិសិដ្ឋនេះត្រូវបានគេជឿថាត្រូវបានដាក់ចេញដោយផ្អែកលើគោលគំនិតឥណ្ឌាបុរាណនៃសកលលោក។ កោះកេរបានបង្ហាញពីការរៀបចំទីក្រុង និងលក្ខណៈស្ថាបត្យកម្មមិនធម្មតា ដែលជាលទ្ធផលចម្បងនៃការរួមបញ្ចូលគ្នានៃមហិច្ឆិតានយោបាយដ៏ធំរបស់ព្រះបាទជ័យវរ្ម័នទី៤ និងការច្នៃប្រឌិតដ៏អស្ចារ្យពីរដែលជួយសម្រេចមហិច្ឆតានេះ៖ ការបង្ហាញសិល្បៈនៃរចនាបថកោះកេរ និងការស្ថាបនា។ បច្ចេកវិទ្យាដោយប្រើប្លុកថ្ម monolithic ដ៏ធំ។ ថ្វីត្បិតតែមានអាយុកាលខ្លីជារាជធានី ហើយដូច្នេះគ្រាន់តែជាការជ្រៀតជ្រែកក្នុងប្រវត្តិសាស្ត្រខ្មែរក៏ដោយ ការច្នៃប្រឌិតទាំងនេះមានឥទ្ធិពលយ៉ាងជ្រាលជ្រៅ និងយូរអង្វែងលើការសាងសង់ទីក្រុង និងការបង្ហាញសិល្បៈក្នុងតំបន់។
            <br><br>
            <a href="https://whc.unesco.org/en/list/1667/" style="color: #d30000; text-decoration: none;" target="_blank">ប្រភព៖ UNESCO</a>
            </div>
            """
        st.markdown(html_text_KK_KH, unsafe_allow_html=True)

        st.markdown("#### Some information of Koh Ker Temple:")
        html_text_KK_EN = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            The archaeological site of Koh Ker is a sacred urban ensemble of numerous temples and sanctuaries including sculptures, inscriptions, wall paintings, and archaeological remains. Constructed over a twenty-three-year period, it was one of two rival Khmer Empire capitals – the other being Angkor – and was the sole capital from 928 to 944 CE. Established by King Jayavarman IV, his sacred city was believed to be laid out on the basis of ancient Indian religious concepts of the universe. The new city demonstrated unconventional city planning, artistic expression and construction technology, especially the use of very large monolithic stone blocks.

            Koh Ker: Archaeological Site of Ancient Lingapura or Chok Gargyar was a capital of the Khmer Empire between 921 and 944 CE. Partially hidden in a dense broad-leaf forest between the Dangrek and Kulen mountain ranges on a gently sloping hill some eighty kilometres northeast of Angkor, the archaeological site comprises numerous temples and sanctuaries with associated sculptures, inscriptions, and wall paintings, archaeological remains and hydraulic structures.

            Established by King Jayavarman IV in 921 CE, Koh Ker was one of two rival capitals of the Khmer Empire that co-existed between 921 and 928 CE – the other being Angkor – and the sole capital until 944 CE, after which the Empire’s political centre moved back to Angkor. Constructed in a single phase over a twenty-three-year period, the sacred city was believed to be laid out on the basis of ancient Indian concepts of the universe. Koh Ker demonstrated markedly unconventional city planning and architectural features, which were primarily the result of the combination of King Jayavarman IV’s grand political ambition and the two outstanding innovations that helped to materialise this ambition: the artistic expressions of the Koh Ker Style, and the construction technology using very large monolithic stone blocks. Although short-lived as a capital and thus acting only as an interlude in Khmer history, these innovations had a profound and lasting influence on urban construction and artistic expression in the region.
            <br><br>
            <a href="https://whc.unesco.org/en/list/1667/" style="color: #d30000; text-decoration: none;" target="_blank">Source: UNESCO</a>
            </div>
            """
        st.markdown(html_text_KK_EN, unsafe_allow_html=True)
    elif predicted_class == 'Prasat Sambor Prei Kuk':
        st.sidebar.warning(result_message)
        st.markdown("## ប្រាសាទសំបូរព្រៃគុក - Prasat Sambo Prei Kuk")
        st.markdown("#### ពត៌មានខ្លះៗនៃប្រាសាទសំបូរព្រៃគុក៖")
        html_text_PSPK_KH = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            ប្រាសាទសំបូរព្រៃគុក “ប្រាសាទសំបូរព្រៃគុក” ជាភាសាខ្មែរ ត្រូវបានកំណត់ថាជា ឥសានបុរៈ រាជធានីនៃអាណាចក្រចេនឡា ដែលបានរីកដុះដាលនៅចុងសតវត្សទី៦ និងដើមសតវត្សទី៧ នៃគ.ស។ អចលនទ្រព្យនេះរួមមានប្រាសាទជាងមួយរយ ដែលក្នុងចំណោមប្រាសាទចំនួនដប់ មានរាងប្រាំបីជ្រុង ដែលជាគំរូតែមួយគត់នៃប្រភេទរបស់ពួកគេនៅអាស៊ីអាគ្នេយ៍។ គ្រឿងតុបតែងថ្មភក់នៅក្នុងទីតាំងគឺជាលក្ខណៈនៃសទ្ទានុក្រមតុបតែងមុនអង្គរ ដែលគេស្គាល់ថាជារចនាបថសំបូរព្រៃគុក។ ធាតុ​មួយ​ចំនួន​នេះ រួម​មាន​ធ្នឹម ជើង​ទម្រ និង colonnades គឺជា​ស្នាដៃ​ពិត។ សិល្បៈ និងស្ថាបត្យកម្មដែលបានអភិវឌ្ឍនៅទីនេះបានក្លាយទៅជាគំរូសម្រាប់ផ្នែកផ្សេងទៀតនៃតំបន់ និងជាមូលដ្ឋានគ្រឹះសម្រាប់រចនាបថខ្មែរតែមួយគត់នៅសម័យអង្គរ។
            តំបន់ប្រាសាទសំបូរព្រៃគុក គឺជាផ្នែកមួយនៃសំណល់នៃប្រាសាទឥសានបុរៈបុរាណ "ប្រាសាទនៅក្នុងព្រៃខៀវស្រងាត់" ដែលជារាជធានីនៃអាណាចក្រចេនឡា ដែលបានរីកដុះដាលលើអាស៊ីអាគ្នេយ៍ជាច្រើននៅចុងសតវត្សទី 6 និងដើមសតវត្សទី 7 នៃគ.ស ហើយមានស្ថាបត្យកម្ម សមិទ្ធិផលបានចាក់គ្រឹះសម្រាប់អាណាចក្រខ្មែរជំនាន់ក្រោយ។ តំបន់ប្រាសាទដ៏ធំទូលាយនៃផ្ទៃដី 840 ហិចតា ស្ថិតនៅភាគខាងកើតនៃសំណល់នៃទីក្រុង mowed ហើយត្រូវបានតភ្ជាប់ទៅស្ទឹងសែន និងកំពង់ផែដែលអាចធ្វើទៅបាននៃ Ishanapura ដោយផ្លូវដីចំនួន 3 ប្រវែងចន្លោះពី 600 ទៅ 700 ម៉ែត្រ។

            នៅក្នុងតំបន់ប្រាសាទ ក្រុមដ៏អស្ចារ្យនៃប្រាសាទឥដ្ឋចំនួន 186 ដែលត្រូវបានដុតដោយថ្មភក់លម្អិត ឆ្លុះបញ្ចាំងពីការណែនាំអំពីគំនិតបច្ចេកទេស និងខាងវិញ្ញាណនៃសាសនាហិណ្ឌូ Hariharan និង Sakabrahmana ពីប្រទេសឥណ្ឌា និងពែរ្សរៀងៗខ្លួន និងការបញ្ចូលគ្នាជាលទ្ធផលនៃវត្ថុទាំងនេះជាមួយនឹងធាតុនិស្ស័យ និងពុទ្ធសាសនាដែលបានបង្កើត រចនាបថសិល្បៈប្រាសាទសំបូរព្រៃគុកតែមួយគត់ ដែលក្រោយមកបានផ្សព្វផ្សាយពីរចនាបថខ្មែរដែលបានអភិវឌ្ឍនៅអង្គរ។ សិលាចារឹកជាភាសាសំស្រ្កឹត និងខ្មែរបុរាណនៅលើប្រាសាទមួយចំនួន ឆ្លុះបញ្ចាំងពីការទទួលយក "ព្រះ-ស្តេច" នៅក្នុងរដ្ឋកណ្តាល ខណៈខ្លះទៀតកត់ត្រាអំពីសកម្មភាពប្រាសាទ ព្រះនាមស្តេច និងបុគ្គលដទៃទៀត ព័ត៌មានលម្អិតអំពីជីវិតសាសនា និងនយោបាយ និងផ្តល់យោបល់។ ព្រំដែនរួមនៃចក្រភព។ ការផ្តល់ជំនួយដល់ប្រាសាទគឺជាសញ្ញាដំបូងនៃការនិទានរឿងដែលមើលឃើញនៅក្នុងការតុបតែងប្រាសាទដែលហួសពីការបង្ហាញបុរាណនៃអាទិទេពនៅក្នុងមេដាយតូចៗ ឬរូបចម្លាក់តូចៗជិះសត្វទេវកថា។

            មានប្រាសាទសំខាន់ៗចំនួនបីគឺ ប្រាសាទយាយយ៉ (ក្រុមខាងត្បូង) ប្រាសាទតាវ (ក្រុមកណ្តាល) ប្រាសាទសំបូរ (ក្រុមខាងជើង រួមទាំងក្រុមប្រាសាទសណ្តាន់ និងប្រាសាទបុស្សរាម)។ អគារនីមួយៗមានប្រាង្គកណ្តាលនៅលើវេទិកាលើកកំពស់ ហ៊ុំព័ទ្ធដោយប៉មតូចៗ និងសំណង់ផ្សេងៗទៀត ហើយត្រូវបានរុំព័ទ្ធដោយឥដ្ឋរាងការ៉េ និង/ឬជញ្ជាំងថ្មបាយក្រៀម ពីរសម្រាប់ក្រុមកណ្តាល និងខាងត្បូង ប៉ុន្តែបីសម្រាប់ប្រាសាទសំបូរ ដែលមានជញ្ជាំងខាងក្រៅនីមួយៗលាតសន្ធឹងដល់ ៣៨៩។ ម៉ែត្រ។ ក្រុមទាំងបីនេះមានប្រាសាទបុគ្គលចំនួន 125 ដែលមានប្រាសាទ និងសំណង់ចំនួន 46 ផ្សេងទៀតនៅក្នុងតំបន់ជុំវិញរួមមានក្រុមប្រាសាទត្រពាំងរលាប និងក្រុមប្រាសាទគោកត្រែង។ នៅភាគខាងជើង តំបន់រណបនៃប្រាសាទចំនួន 16 នៅក្នុងក្រុមប្រាសាទស្រែប្រាំង និងប្រាសាទរមាសរមាស បង្ហាញពីការផ្លាស់ប្តូរស្ថាបត្យកម្មពីរចនាបថស្ថាបត្យកម្មហ្សេនឡា (ចេនឡា) សម័យមុន ទៅជាប្រាសាទសំបូរព្រៃគុក។ នៅក្នុងតំបន់នេះ ស្រទាប់បុរាណវត្ថុវិទ្យាយ៉ាងទូលំទូលាយត្រូវបានសាងសង់ឡើងលើគ្នាទៅវិញទៅមក នៅតែត្រូវបានបិទបាំង។

            ប្រាសាទនានាត្រូវបានសាងសង់ក្នុងទម្រង់ផ្សេងៗគ្នា ការកំណត់រចនាសម្ព័ន្ធ និងទំហំ ប៉ុន្តែអ្វីដែលពិសេសនោះគឺប្រាសាទ 11 octagonal ដែលត្រូវបានរចនាឡើងស្របតាមគោលការណ៍ទូទៅនៃសៀវភៅណែនាំស្ថាបត្យកម្មឥណ្ឌាបុរាណ (ទោះបីជាមិនមានគំរូមុនរបស់ឥណ្ឌាក៏ដោយ) ។ ទាំងនេះត្រូវបានគេមើលឃើញថាតំណាងឱ្យវិមាន octagonal ហោះរបស់ព្រះឥន្ទ្រឬ Vimana Trivishtapa, ឋានសួគ៌នៃព្រះឥន្ទ្រនិងនៃ 33 ព្រះ។ ជញ្ជាំងខាងក្រៅត្រូវបានតុបតែងដោយរូបចម្លាក់ហិណ្ឌូ ហើយនៅក្នុងប្រាសាទចំនួនប្រាំមួយមានរូបចម្លាក់ដ៏អស្ចារ្យនៃវិមានហោះហើរ។

            អគារសាសនាដ៏ធំទូលាយ និងរចនាសម្ព័ន្ធបន្ថែមរបស់វា រួមជាមួយនឹងលក្ខណៈធារាសាស្ត្រចំនួន 102 បង្ហាញពីសមិទ្ធិផលក្នុងការរៀបចំផែនការ ភាពប៉ិនប្រសប់ផ្នែកបច្ចេកទេស ការប្រតិបត្តិ និងការគ្រប់គ្រងធនធានដែលមិនធ្លាប់មានពីមុនមកនៅក្នុងតំបន់អាស៊ីអាគ្នេយ៍។
            <br><br>
            <a href="https://whc.unesco.org/en/list/1532/" style="color: #d30000; text-decoration: none;" target="_blank">ប្រភព៖ UNESCO</a>
            </div>
            """
        st.markdown(html_text_PSPK_KH, unsafe_allow_html=True)

        st.markdown("#### Some information of Sambo Prei Kuk Temple:")
        html_text_PSPK_EN = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            The archaeological site of Sambor Prei Kuk, “The temple in the richness of the forest” in the Khmer language, has been identified as Ishanapura, the capital of the Chenla Empire that flourished in the late 6th and early 7th centuries AD. The property comprises more than a hundred temples, ten of which are octagonal, unique specimens of their genre in South-East Asia. Decorated sandstone elements in the site are characteristic of the pre-Angkor decorative idiom, known as the Sambor Prei Kuk Style. Some of these elements, including lintels, pediments and colonnades, are true masterpieces. The art and architecture developed here became models for other parts of the region and lay the ground for the unique Khmer style of the Angkor period.
            Sambor Prei Kuk Temple Zone is part of the remains of ancient Ishanapura "the temple in the lush forest", which was the capital of the Chenla Empire that flourished over much of Southeast Asia in the late 6th and early 7th centuries AD, and whose architectural achievements laid the foundations for those of the later Khmer Empire. The extensive Temple Zone of 840 hectares lies to the east of the remains of the moated city and is linked to the river Stung Sen and a possible harbour of Ishanapura by three earthen causeways between 600 and 700 metres in length.

            Within the Temple Zone, an outstanding ensemble of 186 fired brick temples with sandstone detailing reflects the introduction of technical and spiritual ideas of the Hindu Hariharan and Sakabrahmana cults from India and Persia respectively and the resulting convergence of these with animist and Buddhist elements that produced the unique Sambor Prei Kuk artistic style, which later heralded the Khmer style developed in Angkor. Inscriptions in Sanskrit and old Khmer on some of the temples reflect the adoption of a “God-King” in the centralized state, while others record temple activities, the names of kings and other individuals, details of religious and political life, and suggest the overall boundaries of the empire. The temple reliefs are the first signs of visual narratives in temple decoration which go beyond the earlier standard heraldic displays of deities in small medallions or small figures riding mythological animals.

            There are three main temple complexes of Prasat Yeai (Southern Group), Prasat Tao (Central Group), Prasat Sambor (Northern Group, including the Prasat Sandan Group and Prasat Bos Ream). Each has a central tower on a raised platform surrounded by smaller towers and other structures, and are enclosed by square brick and/or laterite walls, two for the central and south groups but three for the Prasat Sambor complex with each outer wall extending to 389 metres. These three groups contain 125 individual temples with 46 other temples and structures in the surrounding area including the Prasat Trapeang Ropeak and Prasat Kuok Troung groups. To the north, a satellite zone of 16 temples in the Prasat Srei Krup Leak and Prasat Robang Romeas groups display the architectural transition from the earlier Zhenla (Chenla) architectural style to that of Sambor Prei Kuk. In this area extensive archaeology layers built upon each other remain to be uncovered.

            The temples are constructed in a variety of shapes, configurations, and sizes, but of special note are 11 octagonal temples, designed in accordance with the general principles of the ancient Indian Manuals of Architecture, (although with no known Indian precedent). These are seen to represent the flying octagonal palace of Indra or Vimana Trivishtapa, the heaven of Indra and of 33 gods. The outside walls are decorated with Hindu iconography, and in six temples there are exquisite sculptural depictions of flying palaces.

            The extensive ensemble of religious buildings and their ancillary structures together with 102 hydraulic features display achievements in planning, technical ingenuity, execution, and resource management not previously seen in Southeast Asia.
            <br><br>
            <a href="https://whc.unesco.org/en/list/1532/" style="color: #d30000; text-decoration: none;" target="_blank">Source: UNESCO</a>
            </div>
            """
        st.markdown(html_text_PSPK_EN, unsafe_allow_html=True)
    elif predicted_class == 'Preah_Vihear':
        st.sidebar.warning(result_message)
        st.markdown("## ប្រាសាទព្រះវិហារ - Preah Vihear")
        st.markdown("#### ពត៌មានខ្លះៗនៃប្រាសាទព្រះវិហារ៖")
        html_text_PVH_KH = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            ប្រាសាទព្រះវិហារ ជាសំណង់ស្ថាបត្យកម្មដ៏វិសេសវិសាលនៃទីសក្ការៈជាបន្តបន្ទាប់ ដែលភ្ជាប់គ្នាដោយប្រព័ន្ធក្រាលកៅស៊ូ និងជណ្ដើរលើអ័ក្សប្រវែង ៨០០ ម៉ែត្រ គឺជាស្នាដៃឆ្នើមនៃស្ថាបត្យកម្មខ្មែរ ទាក់ទងនឹងផែនការ ការតុបតែង និងទំនាក់ទំនងជាមួយទេសភាពដ៏អស្ចារ្យ។

            ប្រាសាទព្រះវិហារស្ថិតនៅលើគែមខ្ពង់រាបដែលគ្របដណ្ដប់លើវាលទំនាបនៃប្រទេសកម្ពុជា ប្រាសាទព្រះវិហារត្រូវបានឧទ្ទិសដល់ព្រះសិវៈ។ ប្រាសាទនេះត្រូវបានផ្សំឡើងដោយទីសក្ការៈជា
            បន្តបន្ទាប់ដែលតភ្ជាប់ដោយប្រព័ន្ធផ្លូវ និងជណ្តើរលើអ័ក្សប្រវែង ៨០០ ម៉ែត្រ ហើយមានអាយុកាលតាំងពីពាក់កណ្តាលទីមួយនៃសតវត្សទី ១១ នៃគ.ស។ យ៉ាង​ណា​ក៏​ដោយ ប្រវត្តិ​ដ៏​ស្មុគ​ស្មាញ​របស់​វា​អាច​ត្រូវ​បាន​គេ​កត់​សម្គាល់​នៅ​សតវត្ស​ទី​៩ ដែល​ជា​ពេល​ដែល​អាស្រម​នេះ​ត្រូវ​បាន​បង្កើត​ឡើង។ គេហទំព័រនេះត្រូវបានរក្សាទុកយ៉ាងល្អ ជាពិសេសដោយសារតែទីតាំងដាច់ស្រយាលរបស់វា។ ទីតាំងនេះគឺពិសេសសម្រាប់គុណភាពនៃស្ថាបត្យកម្មរបស់វា ដែលប្រែប្រួលទៅតាមបរិយាកាសធម្មជាតិ និងមុខងារសាសនារបស់ប្រាសាទ ក៏ដូចជាគុណភាពពិសេសនៃគ្រឿងតុបតែងថ្មចម្លាក់របស់វា។

            ភាពត្រឹមត្រូវ ទាក់ទងនឹងរបៀបដែលអគារ និងសម្ភារៈបង្ហាញយ៉ាងច្បាស់ពីតម្លៃនៃអចលនទ្រព្យត្រូវបានបង្កើតឡើង។ លក្ខណៈនៃទ្រព្យរួមមានប្រាសាទ; បូរណភាពនៃទ្រព្យសម្បត្តិមានដល់កម្រិតមួយត្រូវបានសម្របសម្រួលដោយអវត្តមាននៃផ្នែកនៃ promontory ពីបរិវេណនៃទ្រព្យសម្បត្តិ។ វិធានការការពារប្រាសាទ ក្នុងលក្ខខណ្ឌនៃការការពារផ្លូវច្បាប់គឺគ្រប់គ្រាន់។ វឌ្ឍនភាពដែលបានធ្វើឡើងក្នុងការកំណត់ប៉ារ៉ាម៉ែត្រនៃផែនការគ្រប់គ្រងចាំបាច់ត្រូវបញ្ចូលទៅក្នុង
            ផែនការគ្រប់គ្រងពេញលេញដែលត្រូវបានអនុម័ត។
            <br><br>
            <a href="https://whc.unesco.org/en/list/1224/" style="color: #d30000; text-decoration: none;" target="_blank">ប្រភព៖ UNESCO</a>
            </div>
            """
        st.markdown(html_text_PVH_KH, unsafe_allow_html=True)

        st.markdown("#### Some information of Preah Vihear Temple:")
        html_text_PVH_EN = """
            <div style="
            padding: 10px 20px;
            background-color: #eef6f9;
            border-left: 6px solid #2b6cb0;
            border-radius: 4px;
            color: #2b6cb0;
            margin-bottom: 25px;
            text-align: justify;">
            The Temple of Preah Vihear, a unique architectural complex of a series of sanctuaries linked by a system of pavements and staircases on an 800 metre long axis, is an outstanding masterpiece of Khmer architecture, in terms of plan, decoration and relationship to the spectacular landscape environment.

            Situated on the edge of a plateau that dominates the plain of Cambodia, the Temple of Preah Vihear is dedicated to Shiva. The Temple is composed of a series of sanctuaries linked by a system of pavements and staircases over an 800 metre long axis and dates back to the first half of the 11th century AD. Nevertheless, its complex history can be traced to the 9th century, when the hermitage was founded. This site is particularly well preserved, mainly due to its remote location. The site is exceptional for the quality of its architecture, which is adapted to the natural environment and the religious function of the temple, as well as for the exceptional quality of its carved stone ornamentation.

            Authenticity, in terms of the way the buildings and their materials express well the values of the property, has been established. The attributes of the property comprise the temple complex; the integrity of the property has to a degree been compromised by the absence of part of the promontory from the perimeter of the property. The protective measures for the Temple, in terms of legal protection are adequate; the progress made in defining the parameters of the Management Plan needs to be consolidated into an approved, full Management Plan.
            <br><br>
            <a href="https://whc.unesco.org/en/list/1224/" style="color: #d30000; text-decoration: none;" target="_blank">Source: UNESCO</a>
            </div>
            """
        st.markdown(html_text_PVH_EN, unsafe_allow_html=True)

# Hide deprecation warning for file uploader
st.set_option('deprecation.showfileUploaderEncoding', False)