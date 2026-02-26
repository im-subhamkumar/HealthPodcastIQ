import { SummaryResult, SummaryLength, EpisodeSequenceResult } from '../types';
import { GoogleGenAI, Type } from "@google/genai";

// --- CLIENT-SIDE FALLBACK CONFIGURATION ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- MOCK DATA FOR CLIENT-SIDE FALLBACK ---
const singlePodcastTranscript = `
Host: Welcome everyone to 'The Science of Health & Fitness.' Today, we're doing a deep dive, answering all the big questions. Let's start with the absolute foundation: What is the most critical factor for muscle growth?
Expert: Great question. It boils down to two things: progressive overload and adequate protein. You need to challenge your muscles to grow, and you need to give them the building blocks to do so. That means lifting heavier over time and eating enough protein, ideally 1.6 to 2.2 grams per kilogram of body weight. The science on this, from researchers like Schoenfeld, is crystal clear. Progressive overload is the stimulus.
Host: Perfect. Let's move to another area people get wrong: sleep. How does sleep impact fitness and weight loss?
Expert: It's non-negotiable. Poor sleep is a hormonal disaster. It spikes cortisol, your primary stress hormone, which encourages fat storage and can even break down muscle. We have solid data from studies, like one from Leproult in 1997, showing this direct link. It crushes growth hormone release and throws your hunger hormones, ghrelin and leptin, completely out of whack, making you crave junk food.
Host: That leads to the classic debate: Is cardio or weightlifting better for fat loss?
Expert: The best answer is both. It's not a competition. Weightlifting is your long-term investment; it builds muscle, which is metabolically active tissue. More muscle means a higher basal metabolic rate, or BMR. You burn more calories just existing. Cardio, on the other hand, is a great tool for burning calories in the moment. Combining them is the optimal strategy for sustainable fat loss. Don't fall for myths like 'drinking lemon water melts belly fat' – that's physiologically impossible.
Host: Let's talk about what fuels these workouts. How important is hydration for performance?
Expert: Critically important. The American College of Sports Medicine has shown that even a 2% drop in body weight from dehydration can tank your performance. Your blood volume drops, your heart has to work harder, and your focus just disappears. The old '8 glasses a day' rule is a decent starting point, but it's not a hard-and-fast rule; needs vary.
Host: And what about supplements? People spend a fortune on them. Are expensive supplements like BCAAs necessary?
Expert: For 99% of people, no. If your protein intake is sufficient, you're already getting all the BCAAs you need. A 2021 review by Plotkin confirmed that isolated BCAA supplementation doesn't really add a benefit for muscle growth if you're eating enough protein. Save your money, buy good food.
Host: I hear a lot about the 'mind-muscle connection'. Is it real?
Expert: Absolutely. It's about internal focus. Studies from researchers like Brad Schoenfeld have used EMG to show that when you consciously think about the muscle you're working, you can actually increase its electrical activity and activation. It's a powerful tool, especially for isolation exercises.
Host: We've covered training, but what about not training? How important are rest days for recovery?
Expert: They are arguably when you actually get stronger. Exercise is the stimulus that creates micro-tears in the muscle. Rest is when your body repairs those tears, making the muscle bigger and stronger. This is a fundamental concept. Without rest, you're just breaking yourself down, leading to overtraining.
Host: A huge point of confusion is nutrition. Are carbohydrates 'bad' for weight loss?
Expert: Not at all. This is a myth born from fad diets. The first law of thermodynamics always wins: weight loss is about a calorie deficit. Carbs are your body's preferred fuel source for high-intensity activity. Cutting them out entirely is a recipe for poor performance. The key is quality and quantity.
Host: Okay, if BCAAs are out, what is the most effective, science-backed supplement?
Expert: Easy. Besides basic whey protein, it's Creatine Monohydrate. The ISSN, the International Society of Sports Nutrition, has called it the most effective ergogenic supplement available. It's safe, cheap, and has hundreds of studies backing its ability to improve strength and power output.
Host: Let's bust another myth. Does fasted cardio burn more body fat?
Expert: It's technically true that you burn a higher percentage of fat for fuel *during* the session. However, a key meta-analysis in 2014 showed that over a 24-hour period, if calories are controlled, there's no significant difference in total fat loss between fasted and fed cardio. Do what you prefer and can stick with.
Host: What about the mental side? How does chronic stress impact fitness?
Expert: It's a silent progress killer. Chronic stress means chronic high cortisol. As we discussed, that's bad news. It encourages abdominal fat storage, as shown by researchers like Epel, and it's catabolic, meaning it can break down muscle tissue. Managing stress is as important as managing your workouts.
Host: So if you had to pick one, for weight loss, what matters more: diet or exercise?
Expert: For pure weight loss, diet is the undisputed champion. It is vastly easier to create a 500-calorie deficit by not eating a bagel than it is to burn 500 calories on a treadmill. But for overall health and body composition—losing fat while keeping muscle—you need both. You can't out-train a bad diet.
Host: What about dietary fats? Are they important?
Expert: Essential. We went through a low-fat craze that was a huge mistake. Healthy fats from sources like avocados, nuts, and olive oil are critical for hormone production. Cholesterol is the precursor to hormones like testosterone. If your dietary fat is too low, you're compromising your entire endocrine system.
Host: With cardio, there are so many options. Is HIIT better than steady-state cardio, or LISS?
Expert: They're different tools for different jobs. HIIT is incredibly time-efficient and great for your cardiovascular system, but it's also very stressful. LISS is less taxing, can aid recovery, and is great for building an aerobic base. A good program will likely incorporate both. It's not about one being better, but how you use them.
Host: And what about just basic movement? How important are mobility and flexibility?
Expert: Immensely. People chase strength but neglect the foundation. If you don't have the mobility to get into a deep squat safely, you can't effectively train your legs and you risk injury. Mobility work is your injury prevention insurance.
Host: How often should someone be in the gym? How many days a week should you work out?
Expert: It's highly individual, but the science, again from folks like Schoenfeld, points to training a muscle group about twice a week for optimal growth. For most people, that means a well-designed program hitting 3 to 5 workouts per week is a fantastic target.
Host: Can you actually target fat loss? Can I do a thousand crunches to get a six-pack?
Expert: That is the myth of 'spot reduction,' and it is 100% false. You can't pick where you lose fat from. Your body loses it systemically based on genetics. You do crunches to build your ab muscles, but you only see them by losing overall body fat through a calorie deficit.
Host: How much of this is just predetermined? How much do genetics influence fitness results?
Expert: Genetics play a role, for sure. They can influence your metabolism, where you store fat, your muscle fiber type distribution. But they are not a life sentence. They define your ultimate potential, but 99% of us are nowhere near that potential. Hard work and consistency will always beat lazy genetics.
Host: Let's talk about a common social activity. How does alcohol consumption affect fitness?
Expert: It's pretty much a direct negative. A 2014 study by Parr showed it directly suppresses muscle protein synthesis, which is the process of building muscle. It also messes with your sleep quality and is a source of empty calories. It's one of the easiest things to limit to improve your results.
Host: Finally, we focus so much on macros. Are micronutrients—vitamins and minerals—important?
Expert: They are the engine oil. You can have a full tank of gas (macros), but without the oil (micros), the engine seizes. B vitamins for energy, Vitamin D for hormone function, zinc and magnesium for hundreds of processes. A varied diet full of fruits and vegetables is non-negotiable for a body that performs well.
`;

const podcastTranscript1 = `
Host: Welcome everyone to 'The Science of Health & Fitness.' Today, we're doing a deep dive, answering all the big questions. Let's start with the absolute foundation: What is the most critical factor for muscle growth?
Expert: Great question. It boils down to progressive overload and adequate protein. You need to challenge your muscles to grow, and you need to give them the building blocks to do so.
Host: Perfect. Let's move to another area people get wrong: sleep. How does sleep impact fitness and weight loss?
Expert: It's non-negotiable. Poor sleep spikes cortisol, your primary stress hormone, which encourages fat storage and can even break down muscle.
Host: That leads to the classic debate: Is cardio or weightlifting better for fat loss?
Expert: The best answer is both. Weightlifting builds muscle, which boosts your metabolism long-term. Cardio is a great tool for burning calories in the moment. Combining them is the optimal strategy.
Host: I hear a lot about the 'mind-muscle connection'. Is it real?
Expert: Absolutely. It's about internal focus. Studies have used EMG to show that when you consciously think about the muscle you're working, you can increase its electrical activity and activation.
Host: What about not training? How important are rest days for recovery?
Expert: They are arguably when you actually get stronger. Exercise creates micro-tears in the muscle. Rest is when your body repairs those tears, making the muscle bigger and stronger.
Host: With cardio, there are so many options. Is HIIT better than steady-state cardio, or LISS?
Expert: They're different tools. HIIT is time-efficient and great for your cardiovascular system, but it's very stressful. LISS is less taxing, can aid recovery, and is great for building an aerobic base. A good program will likely incorporate both.
Host: Can you actually target fat loss? Can I do a thousand crunches to get a six-pack?
Expert: That is the myth of 'spot reduction,' and it is 100% false. You can't pick where you lose fat from. Your body loses it systemically. You do crunches to build your ab muscles, but you only see them by losing overall body fat.
`;

const podcastTranscript2 = `
Host: Welcome back to 'Nutrition Nerds.' Today we are stripping it all back to basics. First question: what actually IS a calorie?
Expert: It's simply a unit of energy. The food we eat provides energy, measured in calories. To lose weight, you need to expend more energy than you consume.
Host: So that leads us to the big one. What is a calorie deficit?
Expert: A calorie deficit is the state of consuming fewer calories than your body needs to maintain its current weight. This is the fundamental, non-negotiable principle of fat loss. If you are not in a deficit, you will not lose fat. It's the first law of thermodynamics.
Host: Let's talk about the food itself. What are macronutrients?
Expert: They're the three main categories of nutrients you get energy from: protein, carbohydrates, and fats. Each plays a different role in the body.
Host: When it comes to fat loss, why is protein so important?
Expert: Three main reasons. First, it's highly satiating, meaning it keeps you feeling full. Second, it has a higher thermic effect of food, so your body burns more calories digesting it. And third, it's crucial for preserving muscle mass while you're in a calorie deficit.
Host: The eternal question: to achieve a deficit, is it better to eat a low-carb diet or a low-fat diet?
Expert: Honestly, for pure fat loss, it doesn't matter as long as the calorie deficit and protein intake are the same. The best diet is the one you can stick to consistently. Adherence is the most important factor for long-term success.
Host: So how can someone set up a sustainable diet for themselves?
Expert: Start by finding your maintenance calories using an online calculator. Then, create a modest deficit, maybe 300-500 calories. Prioritize protein, eat plenty of vegetables for fiber and micronutrients, and fill the rest with carbs and fats in a way that you enjoy and can sustain.
`;

const getYouTubeVideoId = (url: string): string | null => {
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  return (match && match[2].length === 11) ? match[2] : null;
};

// --- BACKEND API CONFIGURATION ---
const BACKEND_URL = 'http://localhost:8000';

export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: 'GET', signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch (e) {
    return false;
  }
};

export const getHistory = async (): Promise<any[]> => {
  try {
    const res = await fetch(`${BACKEND_URL}/api/history`);
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    return [];
  }
};

// --- CORE FUNCTIONS ---

export const processPodcast = async (source: string | File): Promise<SummaryResult> => {
  let thumbnailUrl: string | undefined = undefined;
  let sourceUrl = '';

  if (typeof source === 'string') {
    sourceUrl = source;
    const videoId = getYouTubeVideoId(source);
    if (videoId) { thumbnailUrl = `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`; }
  } else {
    sourceUrl = 'file_upload';
  }

  // Attempt to use Python Backend First
  try {
    const isBackendOnline = await checkBackendHealth();
    if (isBackendOnline) {
      console.log("Connected to Python Backend. Delegating processing...");

      let response: Response;

      // Handle file upload vs URL
      if (source instanceof File) {
        // Upload file using multipart/form-data
        const formData = new FormData();
        formData.append('file', source);

        response = await fetch(`${BACKEND_URL}/process-podcast-file`, {
          method: 'POST',
          body: formData
        });
      } else {
        // Send URL as JSON
        response = await fetch(`${BACKEND_URL}/process-podcast`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: sourceUrl })
        });
      }

      if (!response.ok) {
        // Try to parse error message from backend
        let errorMessage = "Backend processing failed";
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          } else if (errorData.message) {
            errorMessage = errorData.message;
          }
        } catch (e) {
          // If JSON parsing fails, use status text
          errorMessage = `Backend error (${response.status}): ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const rawText = await response.json();
      // The backend returns a JSON string, or potentially the object depending on implementation.
      // Adjusting parsing logic to be safe.
      const parsedResult = typeof rawText === 'string' ? JSON.parse(rawText) : rawText;
      return { ...parsedResult, thumbnailUrl };
    }
  } catch (err) {
    // If it's a known error from backend, re-throw it
    if (err instanceof Error && err.message !== "Backend processing failed") {
      throw err;
    }
    console.warn("Backend unavailable or failed. Falling back to Client-Side generation.", err);
  }

  // Fallback: Client-Side Gemini
  return processPodcastClientSide(source, thumbnailUrl);
};

export const createEpisodeSequence = async (sources: string[] | File[]): Promise<EpisodeSequenceResult> => {
  // Attempt to use Python Backend First
  try {
    const isBackendOnline = await checkBackendHealth();
    if (isBackendOnline) {
      console.log("Connected to Python Backend. Delegating sequencing...");

      let response: Response;

      if (Array.isArray(sources) && sources.length > 0 && sources[0] instanceof File) {
        // Handle file uploads
        const formData = new FormData();
        (sources as File[]).forEach((file, index) => {
          formData.append('files', file);
        });

        response = await fetch(`${BACKEND_URL}/create-sequence-files`, {
          method: 'POST',
          body: formData
        });
      } else {
        // Handle URLs
        response = await fetch(`${BACKEND_URL}/create-sequence`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sources })
        });
      }

      if (!response.ok) {
        let errorMessage = "Backend sequencing failed";
        try {
          const errorData = await response.json();
          if (errorData.detail) errorMessage = errorData.detail;
        } catch (e) { }
        throw new Error(errorMessage);
      }

      const rawText = await response.json();
      const parsedResult = typeof rawText === 'string' ? JSON.parse(rawText) : rawText;
      return parsedResult;
    }
  } catch (err) {
    if (err instanceof Error && err.message !== "Backend sequencing failed") {
      throw err;
    }
    console.warn("Backend unavailable or failed. Falling back to Client-Side generation.", err);
  }

  // Fallback: Client-Side Gemini (only supports URLs in mock/client-side version usually)
  if (Array.isArray(sources) && sources.length > 0 && sources[0] instanceof File) {
    throw new Error("Client-side sequencing for files is not supported. Please ensure the backend is running.");
  }
  return createEpisodeSequenceClientSide(sources as string[]);
};

// --- CLIENT SIDE IMPLEMENTATIONS (Hidden from main export, used as fallback) ---

const summaryResponseSchema = {
  type: Type.OBJECT,
  properties: {
    title: { type: Type.STRING },
    overallSummary: {
      type: Type.OBJECT, properties: { [SummaryLength.SHORT]: { type: Type.STRING }, [SummaryLength.MEDIUM]: { type: Type.STRING }, [SummaryLength.LONG]: { type: Type.STRING }, }, required: [SummaryLength.SHORT, SummaryLength.MEDIUM, SummaryLength.LONG],
    },
    qaPairs: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT, properties: { id: { type: Type.STRING }, question: { type: Type.STRING }, answers: { type: Type.OBJECT, properties: { [SummaryLength.SHORT]: { type: Type.STRING }, [SummaryLength.MEDIUM]: { type: Type.STRING }, [SummaryLength.LONG]: { type: Type.STRING }, }, required: [SummaryLength.SHORT, SummaryLength.MEDIUM, SummaryLength.LONG], }, claims: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { claim: { type: Type.STRING }, verdict: { type: Type.STRING, enum: ['SUPPORTS', 'REFUTES', 'NEUTRAL'] }, confidence: { type: Type.NUMBER }, explanation: { type: Type.STRING }, }, required: ['claim', 'verdict', 'confidence', 'explanation'], }, }, }, required: ['id', 'question', 'answers', 'claims'],
      },
    },
  },
  required: ['title', 'overallSummary', 'qaPairs'],
};

const processPodcastClientSide = async (source: string | File, thumbnailUrl?: string): Promise<SummaryResult> => {
  console.log("Processing single podcast with Client-Side Gemini API");

  const prompt = `
    You are an expert podcast analyst. Your task is to process the following podcast transcript and generate a structured JSON summary.

    **Instructions:**
    1.  Read the entire transcript carefully.
    2.  Generate a concise and engaging title for the podcast based on the transcript's content.
    3.  Generate three overall summaries of the entire podcast: a short one-liner, a medium paragraph, and a long, detailed paragraph.
    4.  Identify **every single question** asked by the host. Do not skip, merge, or paraphrase the questions. List them exactly as they appear.
    5.  For each question, provide three versions of the answer based ONLY on the information present in the transcript: a short one-sentence answer, a medium two-to-three sentence answer, and a long, comprehensive answer.
    6.  For each Q&A pair, identify up to two factual claims made in the expert's answer. For each claim, determine if the transcript supports, refutes, or is neutral towards it. Provide a brief explanation based on the transcript and a confidence score from 0-100.
    7.  For each Q&A pair, generate a unique ID string in the format 'qa' followed by the question number (e.g., 'qa1', 'qa2').
    8.  Format your entire output as a single JSON object that strictly adheres to the provided schema. Do not include any text, markdown, or explanations outside of the final JSON object.

    **Transcript to Analyze:**
    ---
    ${singlePodcastTranscript}
    ---
  `;

  try {
    const result = await ai.models.generateContent({ model: "gemini-3-pro-preview", contents: prompt, config: { responseMimeType: "application/json", responseSchema: summaryResponseSchema, }, });
    const parsedResult = JSON.parse(result.text.trim()) as Omit<SummaryResult, 'thumbnailUrl'>;
    return { ...parsedResult, thumbnailUrl, };
  } catch (error) {
    console.error("Error processing single podcast with Gemini API:", error);
    if (error instanceof Error && error.message.includes('API_KEY')) { throw new Error("API key is invalid or missing. Please check your configuration."); }
    throw new Error("Failed to process the podcast transcript.");
  }
};

const sequenceResponseSchema = {
  type: Type.OBJECT,
  properties: {
    sequenceTitle: { type: Type.STRING },
    sequenceIntroduction: { type: Type.STRING },
    segments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT, properties: { id: { type: Type.STRING }, title: { type: Type.STRING }, summary: { type: Type.STRING }, sourcePodcast: { type: Type.INTEGER }, keyConcept: { type: Type.STRING }, }, required: ['id', 'title', 'summary', 'sourcePodcast', 'keyConcept'],
      },
    },
  },
  required: ['sequenceTitle', 'sequenceIntroduction', 'segments'],
};

const createEpisodeSequenceClientSide = async (sources: string[]): Promise<EpisodeSequenceResult> => {
  console.log("Creating episode sequence with Client-Side Gemini API");
  const transcripts = [{ id: 1, content: podcastTranscript1 }, { id: 2, content: podcastTranscript2 }];

  const prompt = `
    You are an expert curriculum designer and content synthesizer. Your task is to take the following two podcast transcripts and synthesize them into a single, coherent 'episode sequence'.

    **Instructions:**
    1.  Read and understand both transcripts. Transcript 1 is about exercise science. Transcript 2 is about nutrition fundamentals.
    2.  Break down each transcript into logical segments. Each segment should cover a single core topic or question.
    3.  For each segment, identify the key concept being discussed (e.g., "Calorie Deficit", "Progressive Overload").
    4.  Create a new, unified sequence of these segments from BOTH podcasts. This new sequence must be structured logically to create the best learning experience. Foundational concepts MUST be introduced before more advanced topics that build upon them. For example, a segment defining 'Calorie Deficit' (from Transcript 2) must come before a segment discussing 'Cardio for Fat Loss' (from Transcript 1).
    5.  Generate a new, concise title for each segment in the final sequence.
    6.  Write a brief summary for each segment.
    7.  Generate a compelling overall title and a brief introduction for the complete episode sequence.
    8.  For each segment in the final sequence, you must indicate its original source (either 1 or 2).
    9.  Generate a unique ID for each segment in the format 'seg' followed by a number (e.g., 'seg1', 'seg2').
    10. Format your entire output as a single JSON object that strictly adheres to the provided schema. Do not include any text, markdown, or explanations outside of the final JSON object.

    **Transcripts to Synthesize:**
    ---
    **Transcript 1:**
    ${transcripts[0].content}
    ---
    **Transcript 2:**
    ${transcripts[1].content}
    ---
  `;

  try {
    const result = await ai.models.generateContent({ model: "gemini-3-pro-preview", contents: prompt, config: { responseMimeType: "application/json", responseSchema: sequenceResponseSchema, }, });
    const parsedResult = JSON.parse(result.text.trim()) as EpisodeSequenceResult;
    return parsedResult;
  } catch (error) {
    console.error("Error creating episode sequence with Gemini API:", error);
    if (error instanceof Error && error.message.includes('API_KEY')) { throw new Error("API key is invalid or missing. Please check your configuration."); }
    throw new Error("Failed to process the podcast transcripts.");
  }
};
