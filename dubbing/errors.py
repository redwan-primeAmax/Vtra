class DubbingError(Exception):
    """ডাবিং পাইপলাইনের বেস এক্সেপশন।"""
    pass

class InputValidationError(DubbingError): pass
class AudioExtractionError(DubbingError): pass
class TranscriptionError(DubbingError): pass
class TranslationError(DubbingError): pass
class SynthesisError(DubbingError): pass
class RenderError(DubbingError): pass
