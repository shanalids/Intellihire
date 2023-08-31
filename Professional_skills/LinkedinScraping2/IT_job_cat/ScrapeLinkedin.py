#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
linkedin_profile_url = 'https://www.linkedin.com/in/chanuxbro/'
api_key = 'f-lU6GZbwRXk9GWghrhDQw'
headers = {'Authorization': 'Bearer ' + api_key}

response = requests.get(api_endpoint,
                        params={'url': linkedin_profile_url,'skills': 'include'},
                        headers=headers)


# In[2]:


profile_data = response.json()
profile_data


# In[3]:


profile_data['experiences']


# In[4]:


csv_file = "Linkedin_results.csv"


# In[5]:


results = profile_data['public_identifier'], profile_data['full_name'],profile_data['follower_count'], profile_data['occupation'],profile_data['headline'], profile_data['summary'], profile_data['country_full_name'], profile_data['city'], profile_data['experiences'][0],profile_data['industry'], profile_data['education'], profile_data['languages'], profile_data['accomplishment_honors_awards'],profile_data['accomplishment_patents'], profile_data['accomplishment_courses'], profile_data['accomplishment_projects'], profile_data['accomplishment_test_scores'],profile_data['certifications'], profile_data['connections'], profile_data['recommendations'], profile_data['skills'], profile_data['interests']

results


# In[5]:


profile_data['skills']


# In[6]:


Skills = profile_data['skills']
Skills


# In[9]:


# # Open the file in write mode
# with open('output.txt', 'w') as file:
#     # Iterate over the list
#     for item in Skills:
#         # Write each item to the file
#         file.write(item + '\n')



