# FA-Sentiment-Analysis
Survey was done of artists in the Houston, TX area and sentiment analysis was conducted to help direct strategy. This was done for a non-profit firm that had deployed a survey utilizing the survey monkey platform. Three different short form questions were asked to each respondent and a variety of answers were collected. The objective was to compare survey monkey's default algorithm for text analysis to see if a more useful word cloud could be produced. On top of that, sentiment analysis was conducted in order to better understand the mindset of the artists in Houston, TX.

There are three python files that relate to the three different free response questions. Each question has it's own xlsx file with it's respective responses. There is one xlsx file with all the reviews combined, which was utilized to do text analysis. A final report is uploaded under the Project Summary document which details findings.

AFINN and Google sentiment analysis were conducted on these free text responses to compare the accurracy of each package. AFINN was initially attempted using a website interface as an initial manual scoring method before it was done utilizing the package. In order to do Google sentiment one must have permissions to use the platform. This user access ID has been eliminated from the code and is noted. 

It comes as no surprise that Google Sentiment is a far superior package to utilize and was the best for devising a strategy for this non-profit. 
