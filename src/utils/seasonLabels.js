export function getUpcomingProjectionSeasonLabel(date = new Date()) {
  const normalizedDate = date instanceof Date ? date : new Date(date);
  const startYear = normalizedDate.getFullYear();
  const endYear = String((startYear + 1) % 100).padStart(2, '0');

  return `${startYear}-${endYear}`;
}
