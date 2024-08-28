from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
import logging
from vncorenlp import VnCoreNLP
from pyvi import ViTokenizer
from underthesea import word_tokenize
import vietokenizer
import os
import re
import sys

patterns = {
    "TCVN3"        : [ r'\w­[¬íêîëì]|®[¸µ¹¶·Ê¾»Æ¼½ÌÑÎÏªÕÒÖÓÔÝ×ÞØÜãßäáâ«èåéæç¬íêîëìóïôñòøõùö]', 0],
    "VNI_WIN"      : [r'[öô][ùøïûõ]|oa[ëùøïûõ]|ñ[aoeuôö][äàáåãùøïûõ]', re.IGNORECASE],
    "VIQR"         : [r'u[\+\*]o[\+\*]|dd[aoe][\(\^~\'`]|[aoe]\^[~`\'\.\?]|[uo]\+[`\'~\.\?]|a\([\'`~\.\?]', re.IGNORECASE],
    "UNICODE"      : [r'[Ạ-ỹ]', 0],
    "VISCII"       : [r'\wß[½¾¶þ·Þ]|ð[áàÕäã¤í¢£ÆÇè©ë¨êª«®¬­íì¸ïîóò÷öõô¯°µ±²½¾¶þ·ÞúùøüûÑ×ñØ]', 0],
    "VPS_WIN"      : [r'\wÜ[Ö§©®ª«]|Ç[áàåäãÃí¢¥£¤èËÈëêÍíìÎÌïóòÕõôÓÒ¶°Ö§©®ª«úùøûÛÙØ¿º]', 0],
    "VIETWARE_F"   : [r'\w§[¥ìéíêë]|¢[ÀªÁ¶ºÊÛÂÆÃÄÌÑÍÎ£ÕÒÖÓÔÛØÜÙÚâßãàá¤çäèåæ¥ìéíêëòîóïñ÷ôøõ]', 0],
    "VIETWARE_X"   : [r'[áãä][úöûøù]|à[õòûóô]|[åæ][ïìüíî]', re.IGNORECASE]
    }

TCVN3 = ["Aµ", "A¸", "¢" , "A·", "EÌ", "EÐ", "£" , "I×", "IÝ", "Oß",
			"Oã", "¤" , "Oâ", "Uï", "Uó", "Yý", "µ" , "¸" , "©" , "·" ,
			"Ì" , "Ð" , "ª" , "×" , "Ý" , "ß" , "ã" , "«" , "â" , "ï" ,
			"ó" , "ý" , "¡" , "¨" , "§" , "®" , "IÜ", "Ü" , "Uò", "ò" ,
			"¥" , "¬" , "¦" , "­"  , "A¹", "¹" , "A¶", "¶" , "¢Ê", "Ê" ,
			"¢Ç", "Ç" , "¢È", "È" , "¢É", "É" , "¢Ë", "Ë" , "¡¾", "¾" ,
			"¡»", "»" , "¡¼", "¼" , "¡½", "½" , "¡Æ", "Æ" , "EÑ", "Ñ" ,
			"EÎ", "Î" , "EÏ", "Ï" , "£Õ", "Õ" , "£Ò", "Ò" , "£Ó", "Ó" ,
			"£Ô", "Ô" , "£Ö", "Ö" , "IØ", "Ø" , "IÞ", "Þ" , "Oä", "ä" ,
			"Oá", "á" , "¤è", "è" , "¤å", "å" , "¤æ", "æ" , "¤ç", "ç" ,
			"¤é", "é" , "¥í", "í" , "¥ê", "ê" , "¥ë", "ë" , "¥ì", "ì" ,
			"¥î", "î" , "Uô", "ô" , "Uñ", "ñ" , "¦ø", "ø" , "¦õ", "õ" ,
			"¦ö", "ö" , "¦÷", "÷" , "¦ù", "ù" , "Yú", "ú" , "Yþ", "þ" ,
			"Yû", "û" , "Yü", "ü" , "."]

UNICODE = ["À", "Á", "Â", "Ã", "È", "É", "Ê", "Ì", "Í", "Ò",
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
			"Ỷ", "ỷ", "Ỹ", "ỹ", "."]

VIQR = ["A`" , "A'" , "A^" , "A~" , "E`" , "E'" , "E^" , "I`" , "I'" , "O`" ,
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
			"Y?" , "y?" , "Y~" , "y~" , "\\."]

VNI_WIN = ["AØ", "AÙ", "AÂ", "AÕ", "EØ", "EÙ", "EÂ", "Ì" , "Í" , "OØ",
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
			"YÛ", "yû", "YÕ", "yõ", "."]

VPS_WIN = ["à", "Á", "Â", "‚", "×", "É", "Ê", "µ", "´", "¼",
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
			"›", "›", "Ï", "Ï", "."]

VIETWARE_X = ["AÌ", "AÏ", "Á", "AÎ", "EÌ", "EÏ", "Ã", "Ç", "Ê", "OÌ",
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
			"YÍ", "yí", "YÎ", "yî", "."]

def detectCharset(str_input):
    for pattern in patterns:
        match = re.search(patterns[pattern][0], str_input, patterns[pattern][1])
        if match != None:
            return pattern
    return None

def convert(str_original, target_charset):
    source_charset = detectCharset(str_original)
    map_length = len(source_charset)
    
    for number in range(map_length):
        str_original = str_original.replace(source_charset[number], "::" + str(number) + "::")
    for number in range(map_length):
        str_original = str_original.replace("::" + str(number) + "::", target_charset[number])
    
    return source_charset

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
            raw_output  = rdrsegmenter.tokenize(sentence)
            output = [item for sublist in nested_list for item in sublist]
        case "Pyvi":
            tokenized_sentence = ViTokenizer.tokenize(sentence)
            output = tokenized_sentence.split()
        case "Underthesea":
            raw_output = word_tokenize(sentence)
            output = [word.replace(' ', '_') for word in raw_output]
    
    return output

def handle_encoding(sentence, format_encode):
    return convert(sentence, format_encode)

def remove_stop_word(sentence):
    pass

def xu_ly_dau_cau(sentence):
    pass

def xu_ly_dau_thanh(sentence):
    pass

def remove_html(sentence):
    pass


@app.post("/process")
async def process_text(input: FunctionList):
    list_function = input.dict()["function"]
    output = input.dict()["text"]
    option = input.dict()["option"]
    
    for function in list_function:
        match function:    
            case "tach_tu":
                output = tach_tu(output, option["tach_tu"])
            case "handle_encoding":
                output = handle_encoding(output, option["handle_encoding"])
            case "stop_word":
                output = remove_stop_word(output, option["handle_encoding"])
            case "xu_ly_dau_cau":
                output = dau_cau(output, option["handle_encoding"])
            case "xu_ly_dau_thanh":
                output = handle_encoding(output, option["handle_encoding"])
            case "remove_html":
                output = handle_encoding(output, option["handle_encoding"])
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
