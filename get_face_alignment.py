import face_alignment

def getFaceAlignment(input):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
    return fa.get_landmarks(input)