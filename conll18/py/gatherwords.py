import urllib.request

def url_is_alive(url):
  """
  Checks that a given URL is reachable.
  :param url: A URL
  :rtype: bool
  """
  request = urllib.request.Request(url)
  request.get_method = lambda: 'HEAD'

  try:
      urllib.request.urlopen(request)
      return True
  except urllib.request.HTTPError:
      return False

URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.$lang.300.vec.gz"
URL1 = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.$lang.vec"

commands = ""
langs_seen = {}
with open('../res/w2v.txt') as f:
  for line in f:
    lang = line.strip().split()[1]
    print(lang)
    if lang in langs_seen:
      continue
    langs_seen[lang] = True
    lang = line.strip().split()[1]
    U1 = URL.replace("$lang",lang)
    U2 = URL1.replace("$lang",lang)
    if url_is_alive(U1):
      commands+="wget "+U1+"\n"
    elif url_is_alive(U2):
      commands+="wget "+U2+"\n"
#print(commands.strip())

w = open('../shell/download_fair_vectors.sh', 'w')
w.write(commands.strip())
w.close()


'''
dwn_langs = {}
with open('dwnlink.sh') as f:
  for line in f:
    lang = line.strip().split("/")[-1].split(".")[1]
    dwn_langs[lang] = True


with open('w2v.txt') as f:
  for line in f:
    lang = line.strip().split()[1]
    if lang not in dwn_langs:
      print(lang)
'''
'''
sme - se
grc
grc
fro
kmr
'''
'''
w = open('convert_wv.sh', 'w')
lang_already_seen = {}
with open('dwnlink.sh') as f:
  for line in f:
    content = line.strip().split("/")[-1]
    lang = line.strip().split("/")[-1].split(".")[1]
    if lang in lang_already_seen:
      continue
    lang_already_seen[lang] = True
    if content.endswith('.300.vec.gz'):
      w.write("gunzip < "+content+" > cc."+lang+".300.vec\n")
    else:
      w.write("mv "+content+" cc."+lang+".300.vec\n")
w.write("wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.se.vec\n")
w.write("mv wiki.se.vec cc.sme.300.vec\n")
w.write("wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.el.vec\n") # grc
w.write("mv cc.el.300.vec cc.grc.300.vec\n")
w.write("mv cc.fr.300.vec cc.fro.300.vec\n")
w.close()
'''







