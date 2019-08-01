import traceback
import textract


def get_text(f):
    try:
        return str(textract.process(f, language='eng', encoding='utf-8'), 'utf-8')
    except Exception:
        traceback.print_exc()
        return ''
