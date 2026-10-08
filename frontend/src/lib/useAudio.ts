import { useCallback, useEffect, useRef, useState } from "react";

export interface AudioControls {
  ref: React.RefObject<HTMLAudioElement | null>;
  available: boolean;
  playing: boolean;
  time: number;
  duration: number;
  rate: number;
  toggle: () => void;
  pause: () => void;
  /** Seek and start playing. */
  playFrom: (t: number) => void;
  seek: (t: number) => void;
  setRate: (r: number) => void;
}

export function useAudio(src: string | null, fallbackDuration = 0): AudioControls {
  const ref = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(fallbackDuration);
  const [rate, setRateState] = useState(1);

  useEffect(() => {
    const a = ref.current;
    if (!a) return;
    const onTime = () => setTime(a.currentTime);
    const onMeta = () => isFinite(a.duration) && setDuration(a.duration);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("loadedmetadata", onMeta);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onPause);
    return () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("loadedmetadata", onMeta);
      a.removeEventListener("play", onPlay);
      a.removeEventListener("pause", onPause);
      a.removeEventListener("ended", onPause);
    };
  }, [src]);

  const seek = useCallback((t: number) => {
    const a = ref.current;
    const clamped = Math.max(0, t);
    if (a && src) a.currentTime = clamped;
    setTime(clamped);
  }, [src]);

  const playFrom = useCallback(
    (t: number) => {
      seek(t);
      ref.current?.play().catch(() => {});
    },
    [seek],
  );

  const toggle = useCallback(() => {
    const a = ref.current;
    if (!a || !src) return;
    if (a.paused) a.play().catch(() => {});
    else a.pause();
  }, [src]);

  const pause = useCallback(() => {
    ref.current?.pause();
  }, []);

  const setRate = useCallback((r: number) => {
    if (ref.current) ref.current.playbackRate = r;
    setRateState(r);
  }, []);

  return { ref, available: !!src, playing, time, duration, rate, toggle, pause, playFrom, seek, setRate };
}
