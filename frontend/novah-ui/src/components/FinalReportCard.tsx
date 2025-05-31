interface Props {
  url: string | null;
}

export default function FinalReportCard({ url }: Props) {
  if (!url) return null;
  return (
    <div className="bg-white/10 p-4 rounded-xl mt-4 text-center">
      <a href={url} className="text-accent underline mr-4">Download PDF</a>
      <a href={url} target="_blank" className="text-accent underline">Open in viewer</a>
    </div>
  );
}
