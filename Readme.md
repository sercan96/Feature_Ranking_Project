- "python3 -m venv .venv" => Bu komut proje içinde .venv klasörü oluşturur.
- "source .venv/bin/activate" => Başarılı olursa terminalin başında genelde şöyle bir şey görünür: (.venv) 

Kurulan Kütüphaneler :
 - "pip install pandas numpy scikit-learn matplotlib seaborn openpyxl tensorflow"
 - "pip freeze > requirements.txt" => Projeyi başka bilgisayarda açarsan aynı kütüphaneleri tekrar kurabilirmemizi sağlar.


Tanımlar :
pandas → veri okuma ve tablo işlemleri
numpy → sayısal işlemler
scikit-learn → preprocessing, split, metrics
matplotlib → grafik
seaborn → veri görselleştirme
openpyxl → Excel dosyasını okumak için
tensorflow → CNN ve autoencoder modeli için


Amacımız : Bir veri setindeki feature’ların hangilerinin daha önemli olduğunu CNN veya Autoencoder kullanarak bulmak ve bunları sıralamak.


CNN Çalışma Sistemi :
    CNN’de feature importance → input feature → convolution → activation → output etkisi

    CNN, özellikler (feature’lar) arasındaki yerel ilişkileri ve etkileşimleri öğrenir.
    Birden fazla katman sayesinde hem tekil feature etkisini hem de feature’ların birlikte oluşturduğu örüntüleri (patterns) yakalar.

    Modeli eğitmek için çok katmanlı bir CNN yapısı kullandık. İlk katman feature’ları tek tek işlerken, sonraki katmanlar feature’lar arasındaki ilişkileri ve daha karmaşık pattern’leri öğrenir.

    Eğitim sırasında tüm katmanlar birlikte öğrenir, yani 1. katmanın ağırlıkları aslında 2. ve 3. katmanların etkisiyle şekillenir.

    Ancak feature ranking yaparken tüm katmanlara bakmayız. Çünkü derin katmanlarda feature’lar karışır ve yorumlanamaz hale gelir.

    Bu yüzden sadece ilk katmanın aktivasyonlarını kullanarak, hangi feature’ın daha güçlü sinyal ürettiğini ölçer ve buna göre sıralama yaparız.

    Tüm katmanlar öğrenme sürecine katkı verir, ancak feature importance doğrudan ilk katmandan çıkarılır.

    Bu projede önce veriyi yükleyip CNN modelini kurduk. Model, birden fazla convolution katmanı kullanarak sınıflandırma problemini öğrendi. Eğitim tamamlandıktan sonra feature importance’ı activations’tan değil, doğrudan ilk convolution katmanının kernel ağırlıklarından çıkardık. İlk conv katmanının kernel matrisi feature × filter biçiminde tabloya dönüştürüldü. Ardından her feature için filtreler arasındaki maksimum ve ortalama mutlak ağırlık hesaplandı. Bu skorlar kullanılarak feature’lar büyükten küçüğe sıralandı ve feature ranking dosyaları oluşturuldu.

AutoEncoder Çalışma Mantığı :

    x→Encoder→z→Decoder→x^ (Mantığı)
    minimize ∣∣x−x^∣∣ (Amaç)

    Autoencoder, veriyi daha düşük boyutlu bir temsile (latent space) sıkıştırıp tekrar orijinaline yakın şekilde yeniden üretmeye çalışır.
    Bu süreçte model, veriyi en iyi temsil eden özellikleri öğrenir.
    
    Input: orijinal feature vektörü
    Encoder: veriyi daha küçük temsile indirir
    Bottleneck: sıkıştırılmış özet temsil
    Decoder: bu özeti kullanıp giriş veriyi tekrar üretmeye çalışır

    Örnek:
        girişte 30 feature var
        encoder bunu 16’ya indirir
        sonra 8’e indirir
        decoder tekrar 16’ya çıkarır
        sonra yeniden 30 feature üretir

    Burada modelin amacı sınıf tahmini değil,input’u yeniden üretmek

FARKI (CNN - AUTOENCODER)
    CNN
        supervised
        sınıflandırma öğrenir
        label kullanır
        ranking, sınıf ayırmaya katkı açısından yorumlanabilir

    Autoencoder
        unsupervised
        yeniden üretim öğrenir
        label kullanmaz
        ranking, veri yapısını temsil etme açısından yorumlanabilir

Bu çok önemli. Çünkü Autoencoder ranking’i şu anlama gelir:

    “Bu feature verinin yapısını yeniden kurmak için ne kadar önemli?”

CNN ranking’i ise daha çok şunu söyler:

    “Bu feature sınıflandırma için ne kadar etkili?

