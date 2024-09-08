from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
import logging
from vncorenlp import VnCoreNLP
from pyvi import ViTokenizer
from underthesea import word_tokenize
import re

list_unicode = {
    "TCVN3": ["Aµ", "A¸", "¢" , "A·", "EÌ", "EÐ", "£" , "I×", "IÝ", "Oß",
			"Oã", "¤" , "Oâ", "Uï", "Uó", "Yý", "µ" , "¸" , "©" , "·" ,
			"Ì" , "Ð" , "ª" , "×" , "Ý" , "ß" , "ã" , "«" , "â" , "ï" ,
			"ó" , "ý" , "¡" , "¨" , "§" , "®" , "IÜ", "Ü" , "Uò", "ò" ,
			"¥" , "¬" , "¦" , "<00ad>", "A¹", "¹" , "A¶", "¶" , "¢Ê", "Ê" ,
			"¢Ç", "Ç" , "¢È", "È" , "¢É", "É" , "¢Ë", "Ë" , "¡¾", "¾" ,
			"¡»", "»" , "¡¼", "¼" , "¡½", "½" , "¡Æ", "Æ" , "EÑ", "Ñ" ,
			"EÎ", "Î" , "EÏ", "Ï" , "£Õ", "Õ" , "£Ò", "Ò" , "£Ó", "Ó" ,
			"£Ô", "Ô" , "£Ö", "Ö" , "IØ", "Ø" , "IÞ", "Þ" , "Oä", "ä" ,
			"Oá", "á" , "¤è", "è" , "¤å", "å" , "¤æ", "æ" , "¤ç", "ç" ,
			"¤é", "é" , "¥í", "í" , "¥ê", "ê" , "¥ë", "ë" , "¥ì", "ì" ,
			"¥î", "î" , "Uô", "ô" , "Uñ", "ñ" , "¦ø", "ø" , "¦õ", "õ" ,
			"¦ö", "ö" , "¦÷", "÷" , "¦ù", "ù" , "Yú", "ú" , "Yþ", "þ" ,
			"Yû", "û" , "Yü", "ü" , "."],
    "UNICODE": ["À", "Á", "Â", "Ã", "È", "É", "Ê", "Ì", "Í", "Ò",
			"Ó", "Ô", "Õ", "Ù", "Ú", "Ý", "à", "á", "â", "ã",
			"è", "é", "ê", "ì", "í", "ò", "ó", "ô", "õ", "ù",
			"ú", "ý", "Ă", "ă", "Đ", "đ", "Ĩ", "ĩ", "Ũ", "ũ",
			"Ơ", "ơ", "Ư", "ư", "Ạ", "ạ", "Ả", "ả", "Ấ", "ấ",
			"Ầ", "ầ", "Ẩ", "ẩ", "Ẫ", "ẫ", "Ậ", "ậ", "Ắ", "ắ",
			"Ằ", "ằ", "Ẳ", "ẳ", "Ẵ", "ẵ", "Ặ", "ặ", "Ẹ", "ẹ",
			"Ẻ", "ẻ", "Ẽ", "ẽ", "Ế", "ế", "Ề", "ề", "Ể", "ể",
			"Ễ", "ễ", "Ệ", "ệ", "Ỉ", "ỉ", "Ị", "ị", "Ọ", "ọ",
			"Ỏ", "ỏ", "Ố", "ố", "Ồ", "ồ", "Ổ", "ổ", "Ỗ", "ỗ",
			"Ộ", "ộ", "Ớ", "ớ", "Ờ", "ờ", "Ở", "ở", "Ỡ", "ỡ",
			"Ợ", "ợ", "Ụ", "ụ", "Ủ", "ủ", "Ứ", "ứ", "Ừ", "ừ",
			"Ử", "ử", "Ữ", "ữ", "Ự", "ự", "Ỳ", "ỳ", "Ỵ", "ỵ",
			"Ỷ", "ỷ", "Ỹ", "ỹ", "."],
    "VIQR": ["A`" , "A'" , "A^" , "A~" , "E`" , "E'" , "E^" , "I`" , "I'" , "O`" ,
			"O'" , "O^" , "O~" , "U`" , "U'" , "Y'" , "a`" , "a'" , "a^" , "a~" ,
			"e`" , "e'" , "e^" , "i`" , "i'" , "o`" , "o'" , "o^" , "o~" , "u`" ,
			"u'" , "y'" , "A(" , "a(" , "DD" , "dd" , "I~" , "i~" , "U~" , "u~" ,
			"O+" , "o+" , "U+" , "u+" , "A." , "a." , "A?" , "a?" , "A^'", "a^'",
			"A^`", "a^`", "A^?", "a^?", "A^~", "a^~", "A^.", "a^.", "A('", "a('",
			"A(`", "a(`", "A(?", "a(?", "A(~", "a(~", "A(.", "a(.", "E." , "e." ,
			"E?" , "e?" , "E~" , "e~" , "E^'", "e^'", "E^`", "e^`", "E^?", "e^?",
			"E^~", "e^~", "E^.", "e^.", "I?" , "i?" , "I." , "i." , "O." , "o." ,
			"O?" , "o?" , "O^'", "o^'", "O^`", "o^`", "O^?", "o^?", "O^~", "o^~",
			"O^.", "o^.", "O+'", "o+'", "O+`", "o+`", "O+?", "o+?", "O+~", "o+~",
			"O+.", "o+.", "U." , "u." , "U?" , "u?" , "U+'", "u+'", "U+`", "u+`",
			"U+?", "u+?", "U+~", "u+~", "U+.", "u+.", "Y`" , "y`" , "Y." , "y." ,
			"Y?" , "y?" , "Y~" , "y~" , "\\."],
    "VNI_WIN": ["AØ", "AÙ", "AÂ", "AÕ", "EØ", "EÙ", "EÂ", "Ì" , "Í" , "OØ",
			"OÙ", "OÂ", "OÕ", "UØ", "UÙ", "YÙ", "aø", "aù", "aâ", "aõ",
			"eø", "eù", "eâ", "ì" , "í" , "oø", "où", "oâ", "oõ", "uø",
			"uù", "yù", "AÊ", "aê", "Ñ" , "ñ" , "Ó" , "ó" , "UÕ", "uõ",
			"Ô" , "ô" , "Ö" , "ö" , "AÏ", "aï", "AÛ", "aû", "AÁ", "aá",
			"AÀ", "aà", "AÅ", "aå", "AÃ", "aã", "AÄ", "aä", "AÉ", "aé",
			"AÈ", "aè", "AÚ", "aú", "AÜ", "aü", "AË", "aë", "EÏ", "eï",
			"EÛ", "eû", "EÕ", "eõ", "EÁ", "eá", "EÀ", "eà", "EÅ", "eå",
			"EÃ", "eã", "EÄ", "eä", "Æ" , "æ" , "Ò" , "ò" , "OÏ", "oï",
			"OÛ", "oû", "OÁ", "oá", "OÀ", "oà", "OÅ", "oå", "OÃ", "oã",
			"OÄ", "oä", "ÔÙ", "ôù", "ÔØ", "ôø", "ÔÛ", "ôû", "ÔÕ", "ôõ",
			"ÔÏ", "ôï", "UÏ", "uï", "UÛ", "uû", "ÖÙ", "öù", "ÖØ", "öø",
			"ÖÛ", "öû", "ÖÕ", "öõ", "ÖÏ", "öï", "YØ", "yø", "Î" , "î" ,
			"YÛ", "yû", "YÕ", "yõ", "."],
    "VPS_WIN":["à", "Á", "Â", "‚", "×", "É", "Ê", "µ", "´", "¼",
			"¹", "Ô", "õ", "¨", "Ú", "Ý", "à", "á", "â", "ã",
			"è", "é", "ê", "ì", "í", "ò", "ó", "ô", "õ", "ù",
			"ú", "š", "ˆ", "æ", "ñ", "Ç", "¸", "ï", "¬", "Û",
			"÷", "Ö", "Ð", "Ü", "å", "å", "", "ä", "ƒ", "Ã",
			"„", "À", "…", "Ä", "Å", "Å", "Æ", "Æ", "", "í",
			"¢", "¢", "£", "£", "¤", "¤", "¥", "¥", "Ë", "Ë",
			"Þ", "È", "þ", "ë", "", "‰", "“", "Š", "”", "‹",
			"•", "Í", "Œ", "Œ", "·", "Ì", "Î", "Î", "†", "†",
			"½", "Õ", "–", "Ó", "—", "Ò", "˜", "°", "™", "‡",
			"¶", "¶", "", "§", "©", "©", "Ÿ", "ª", "¦", "«",
			"®", "®", "ø", "ø", "Ñ", "û", "­", "Ù", "¯", "Ø",
			"±", "º", "»", "»", "¿", "¿", "²", "ÿ", "œ", "œ",
			"›", "›", "Ï", "Ï", "."],
    "VIETWARE_X": ["AÌ", "AÏ", "Á", "AÎ", "EÌ", "EÏ", "Ã", "Ç", "Ê", "OÌ",
			"OÏ", "Ä", "OÎ", "UÌ", "UÏ", "YÏ", "aì", "aï", "á", "aî",
			"eì", "eï", "ã", "ç", "ê", "oì", "oï", "ä", "oî", "uì",
			"uï", "yï", "À", "à", "Â", "â", "É", "é", "UÎ", "uî",
			"Å", "å", "Æ", "æ", "AÛ", "aû", "AÍ", "aí", "ÁÚ", "áú",
			"ÁÖ", "áö", "ÁØ", "áø", "ÁÙ", "áù", "ÁÛ", "áû", "ÀÕ", "àõ",
			"ÀÒ", "àò", "ÀÓ", "àó", "ÀÔ", "àô", "ÀÛ", "àû", "EÛ", "eû",
			"EÍ", "eí", "EÎ", "eî", "ÃÚ", "ãú", "ÃÖ", "ãö", "ÃØ", "ãø",
			"ÃÙ", "ãù", "ÃÛ", "ãû", "È", "è", "Ë", "ë", "OÜ", "oü",
			"OÍ", "oí", "ÄÚ", "äú", "ÄÖ", "äö", "ÄØ", "äø", "ÄÙ", "äù",
			"ÄÜ", "äü", "ÅÏ", "åï", "ÅÌ", "åì", "ÅÍ", "åí", "ÅÎ", "åî",
			"ÅÜ", "åü", "UÛ", "uû", "UÍ", "uí", "ÆÏ", "æï", "ÆÌ", "æì",
			"ÆÍ", "æí", "ÆÎ", "æî", "ÆÛ", "æû", "YÌ", "yì", "YÑ", "yñ",
			"YÍ", "yí", "YÎ", "yî", "."],
}


def convert(str_original, source_charset, target_charset):
    if (source_charset == None):
        return "Không thể xác định loại unicde của văn bản"
    
    map_length = len(source_charset)

    for number in range(map_length):
        str_original = str_original.replace(
            source_charset[number], "::" + str(number) + "::"
        )
    for number in range(map_length):
        str_original = str_original.replace(
            "::" + str(number) + "::", target_charset[number]
        )

    return str_original


def load_stopwords(file_path):
    stopwords = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            stopword = line.strip()
            stopwords.add(stopword)
    return stopwords

class FunctionList(BaseModel):
    text: str
    function: list[str]
    option: dict

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tach_tu(sentence, library):
    output = "Thư viện không chính xác!!!"
    match library:
        case "VnCoreNLP":
            rdrsegmenter = VnCoreNLP("http://nginx", 80)
            raw_output = rdrsegmenter.tokenize(sentence)
            output = [item for sublist in raw_output for item in sublist]
        case "Pyvi":
            tokenized_sentence = ViTokenizer.tokenize(sentence)
            output = tokenized_sentence.split()
        case "Underthesea":
            raw_output = word_tokenize(sentence)
            output = [word.replace(" ", "_") for word in raw_output]

    return output


def handle_encoding(sentence, source_unicode, target_unicode):
    if (target_unicode == "VNI Windows"):
        target_unicode = "VNI_WIN"
    elif (target_unicode == "VIETWARE X"):
        target_unicode = "VIETWARE_X"
    elif (target_unicode == "VPS"):
        target_unicode = "VPS_WIN"
        
    if (source_unicode == "VNI Windows"):
        source_unicode = "VNI_WIN"
    elif (source_unicode == "VIETWARE X"):
        source_unicode = "VIETWARE_X"
    elif (source_unicode == "VPS"):
        source_unicode = "VPS_WIN"
    return convert(sentence, list_unicode.get(source_unicode, "UNICODE"), list_unicode.get(target_unicode, "UNICODE")) # type: ignore


def remove_stop_word(sentence):

    stopwords = load_stopwords("stopword.txt")
    result = []
    for word in sentence:
        if word not in stopwords:
            result.append(word)

    return result

def punctuation_handle(sentence):
    # Replace full-width punctuation with half-width punctuation
    output = sentence.replace('，', ',').replace('。', '.').replace('！', '!').replace('？', '?')
    
    # Remove extra spaces around punctuation
    output = re.sub(r'\s([,.!?])', r'\1', output)
    output = re.sub(r'([,.!?])\s+', r'\1 ', output)
    
    # Normalize ellipses
    output = re.sub(r'\s*…\s*', '...', output)
    
    return output

def diacritics_handle(sentence):

    bang_nguyen_am = [
        ["a", "à", "á", "ả", "ã", "ạ", "a"],
        ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
        ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
        ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
        ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
        ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
        ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
        ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
        ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
        ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
        ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
        ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
    ]

    nguyen_am_to_ids = {}
    for i in range(len(bang_nguyen_am)):
        for j in range(len(bang_nguyen_am[i]) - 1):
            nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

    chars = list(sentence)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])  # type: ignore
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = (
                        bang_nguyen_am[5][dau_cau]
                        if chars[1] == "i"
                        else bang_nguyen_am[9][dau_cau]
                    )
            return "".join(chars)
        return sentence

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:
            chars[index] = bang_nguyen_am[x][dau_cau]
            return "".join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
        else:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]

    return "".join(chars)

def remove_html(sentence):
    return re.sub(r"<[^>]*>", "", sentence)


@app.post("/process")
async def process_text(input: FunctionList):
    list_function = input.model_dump()["function"]
    output = input.model_dump()["text"]
    option = input.model_dump()["option"]

    for function in list_function:
        match function:
            case "tách từ":
                output = tach_tu(output, option["tach_tu"])
            case "xử lý encoding":
                output = handle_encoding(output, option["source_encoding"], option["target_encoding"])
            case "loại bỏ hư từ":
                type_output = "string"
                if type(output) is list:
                    list_word = output
                    type_output = "list"
                else:
                    list_word = tach_tu(output, option["tach_tu"])
                output = remove_stop_word(list_word)
                if type_output == "string":
                    output = " ".join([word.replace("_", " ") for word in output])
            case "chuẩn hóa dấu câu":
                output = punctuation_handle(output)
            case "chuẩn hóa dấu thanh":

                type_output = "string"
                if type(output) is list:
                    words = output
                    type_output = "list"
                else:
                    words = output.split()  # type: ignore

                for index, word in enumerate(words):
                    words[index] = diacritics_handle(word)

                if type_output == "string":
                    output = " ".join(words)
                else:
                    output = words
            case "loại bỏ mã HTML":
                if type(output) is list:
                    content_html = " ".join(output)
                    output = [
                        element
                        for element in remove_html(content_html).split(" ")
                        if element != ""
                    ]
                else:
                    output = remove_html(output)
            case "loại bỏ khoảng trắng":
                output = re.sub(r"\s+", " ", str(output))
    if type(output) is list:
        output = " ".join(output)
    
    # handle multiple \n character
    output = re.sub("\n+", "\n", str(output))

    return {"processed_text": output}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
