"""
Stupid query tool by AG
"""
import requests
from bs4 import BeautifulSoup
print 'Running rank query tool...'
head = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.118 Safari/537.36'}
res = requests.get("https://biendata.com/competition/bytecup2016/leaderboard/", headers=head)
res.encoding = 'utf-8'
soup = BeautifulSoup(res.text, "html.parser")
team_multi = soup.find_all("a",class_="team-multi")
rank = 0
print 'Team                             Rank '
for team in team_multi:
    if team.contents[0][0:8] == "MLCLASS_":
        rank = rank + 1
        parent_node = team.parent.parent.parent
        rank_content = parent_node.find("td", class_="rank")
        blank = ' '*(33 - len(team.contents[0]))
        print str(team.contents[0])+str(blank) + str(rank_content.contents[0][21:24])
        if team.contents[0] == "MLCLASS_Virtuoso ":
            print 'Our Team rank in CSCI567 is: ' + str(rank)
            print 'our Team rank in ALL is: ' + str(rank_content.contents[0][21:24])
            break
    else:
        continue
