function Ypred = classify(imds, hardware)

length = imds.ReadSize;
labels = imds.Labels;

for i=1:length
    tmp = readImage(imds,i);
    hardware.predict(tmp);
end

end